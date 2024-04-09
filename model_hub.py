"""
风格转换 - hub
"""

import os
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub


class HubModel():
    def __init__(self):
        output_image_size = 384
        self.content_img_size = (output_image_size, output_image_size)
        self.style_img_size = (256, 256)

        # 加载模型
        hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        self.hub_model = hub.load(hub_handle)

    def crop_center(self, image):
        """将输入图像裁剪成正方形"""

        shape = image.shape
        new_shape = min(shape[1], shape[2])
        offset_y = max(shape[1] - shape[2], 0) // 2  # 计算纵向偏移量，如果高度大于宽度，则需要在垂直方向上进行一定的偏移
        offset_x = max(shape[2] - shape[1], 0) // 2  # 计算横向偏移量

        # 裁剪图像
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)

        return image

    def load_image(self, image_url, image_size=(256, 256), is_local=True):
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

    def style_conver(self, save_path, content_image_url, style_image_url, content_is_local=True, style_is_local=True):
        # 内容图像
        content_image = self.load_image(content_image_url, self.content_img_size, is_local=content_is_local)
        # 风格图像
        style_image = self.load_image(style_image_url, self.style_img_size, is_local=style_is_local)

        # 对风格图像进行池化，降低维度
        style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

        # 用hub model进行风格迁移
        outputs = self.hub_model(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]

        # 显示图像
        self.imshow(stylized_image)

        # 保存图像
        result_image = tf.squeeze(stylized_image, axis=0)  # 移除批处理维度
        result_image = tf.clip_by_value(result_image, 0.0, 1.0)  # 确保像素值在0到1之间
        result_image = result_image * 255  # 将像素值从[0, 1]缩放到[0, 255]
        result_image = tf.cast(result_image, tf.uint8)  # 转换为整数类型

        tf.keras.preprocessing.image.save_img(save_path, result_image)


if __name__ == '__main__':
    hubModel = HubModel()

    content_image_url = 'image/original_img/dog.jpg'  # 原图
    style_image_url = 'image/style_img/style_03.png'  # 风格图

    # 加载图片
    content_image = hubModel.load_image(content_image_url, is_local=True)
    style_image = hubModel.load_image(style_image_url, is_local=True)
    hubModel.imshow(content_image)
    hubModel.imshow(style_image)

    # 风格转换并保存
    content_is_local = True
    style_is_local = True
    save_path = 'result/hub_model/dog_style03_img.png'
    hubModel.style_conver(save_path, content_image_url, style_image_url, content_is_local, style_is_local)
