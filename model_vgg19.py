"""
风格转换 - 自建
使用VGG19模型从原图和风格图中提取特征，并通过梯度下降更新原图，以融合风格图的艺术风格。
"""

import os
import matplotlib.pylab as plt
import tensorflow as tf


class VGG19Model():
    def __init__(self):
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)

        self.content_weight = 1e5  # 控制内容图像的保留程度
        self.style_weight = 1e2  # 控制风格化的程度

        self.epochs = 10
        self.steps_per_epoch = 500

        self.opt = tf.optimizers.Adam(learning_rate=0.005, beta_1=0.99, epsilon=1e-1)

        # 初始创建VGG19模型实例
        self.vgg_model = self.vgg_layers(self.style_layers + self.content_layers)
        self.vgg_model.trainable = False

    def crop_center(self, image):
        """将输入图像裁剪成正方形"""

        shape = image.shape
        new_shape = min(shape[1], shape[2])
        offset_y = max(shape[1] - shape[2], 0) // 2  # 计算纵向偏移量，如果高度大于宽度，则需要在垂直方向上进行一定的偏移
        offset_x = max(shape[2] - shape[1], 0) // 2  # 计算横向偏移量

        # 裁剪图像
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)

        return image

    def load_image(self, image_url, image_size=(512, 512), is_local=True):
        """加载图像"""

        if is_local:
            image_path = image_url
        else:
            image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)

        img = tf.io.decode_image(tf.io.read_file(image_path), channels=3, dtype=tf.float32)[tf.newaxis, ...]
        img = self.crop_center(img)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)

        return img

    def imshow(self, image):
        """显示图像"""

        # 如果图像张量有批处理维度，则移除它
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        plt.axis('off')  # 不显示坐标轴
        plt.show()

    def vgg_layers(self, layer_names):
        """
        取出VGG19模型的指定层参数，用于构建新的模型
        输入图片后，返回content_layers和style_layers的网络层的激活值
        """

        # 加载预训练的VGG19模型，不包含全连接层，并指定imagenet的权重
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False  # 冻结所有层，在训练中不更新权重，只作为特征提取器

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)

        return model

    def gram_matrix(self, input_tensor):
        """
        将vgg_layers()得到的激活值矩阵转换为风格矩阵
        通过对比风格矩阵，用于判断两张图像的风格是否相同
        """

        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / num_locations

    def style_conver_model(self, input_img):
        """构建风格转换模型"""

        vgg = self.vgg_model

        inputs = input_img * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = vgg(preprocessed_input)

        content_outputs = outputs[self.num_style_layers:]
        style_outputs = outputs[:self.num_style_layers]
        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

    def style_conver_loss(self, output_img, content_targets, style_targets):
        """损失函数"""

        content_outputs = output_img['content']  # 图像当前的内容激活值矩阵
        style_outputs = output_img['style']  # 图像当前的风格矩阵

        # 计算内容图像损失
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers

        # 计算风格图像损失
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers

        loss = content_loss + style_loss

        return loss

    def clip_0_1(self, image):
        """修剪图像的数值，确保像素值在0到1之间"""

        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    @tf.function()
    def train_step(self, image, content_targets, style_targets):
        """训练一轮模型"""

        with tf.GradientTape() as tape:
            outputs = self.style_conver_model(image)
            loss = self.style_conver_loss(outputs, content_targets, style_targets)

        # 获取image相对于loss的梯度
        grad = tape.gradient(loss, image)
        # 使用梯度改变image，使其与风格图片越来越像
        self.opt.apply_gradients([(grad, image)])

        image.assign(self.clip_0_1(image))

    def style_conver(self, save_path, content_path, style_path, content_is_local=True, style_is_local=True):
        """风格转换"""

        content_image = self.load_image(content_path, is_local=content_is_local)
        style_image = self.load_image(style_path, is_local=style_is_local)

        content_targets = self.style_conver_model(content_image)['content']  # 图像的内容激活值矩阵
        style_targets = self.style_conver_model(style_image)['style']  # 图像的风格矩阵

        image = tf.Variable(content_image)

        step = 0
        for i in range(self.epochs):
            for j in range(self.steps_per_epoch):
                step += 1
                self.train_step(image, content_targets, style_targets)
            self.imshow(image)
            print("Train step: {}".format(step))

        # 保存图像
        result_img = tf.squeeze(image, axis=0)  # 移除批处理梯度
        result_img = self.clip_0_1(result_img)  # 确保像素值在0到1之间
        result_img = result_img * 255  # 将像素值缩放到[0, 255]
        result_img = tf.cast(result_img, tf.uint8)  # 转换为整数

        tf.keras.preprocessing.image.save_img(save_path, result_img)


if __name__ == '__main__':
    vgg19Model = VGG19Model()

    content_image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'  # 原图
    style_image_url = 'image/style_img/style_02.png'  # 风格图

    # 加载图片
    content_image = vgg19Model.load_image(content_image_url, is_local=False)
    style_image = vgg19Model.load_image(style_image_url, is_local=True)
    vgg19Model.imshow(content_image)
    vgg19Model.imshow(style_image)

    # 风格转换并保存
    content_is_local = False
    style_is_local = True
    save_path = 'result/vgg19_model/dog_style02_img.png'
    vgg19Model.style_conver(save_path, content_image_url, style_image_url, content_is_local, style_is_local)
