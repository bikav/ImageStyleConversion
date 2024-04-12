"""pyinstaller -F -w GUI.py"""

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from hub_model import HubModel
from pixsynth_model import PixSynthModel


class Home():
    def __init__(self, master):
        self.root = master
        self.root.title('Image Style Conversion')
        self.root.geometry('1100x600')
        self.root['background'] = '#000000'

        window(self.root)


class window():
    def __init__(self, master):
        self.master = master
        self.master.config(bg='#554545')
        self.frame = tk.Frame(self.master, width=1100, height=600, bg='#554545')
        self.frame.pack()
        self.content_image_path = None  # 存储原图路径
        self.style_image_path = None  # 存储风格图路径
        self.transparent_image = tk.PhotoImage(width=1, height=1)  # 透明的初始图片，用于隐藏显示框

        # 大标题
        headline = tk.Label(
            self.frame,
            text='Image Style Conversion',
            font=('Times New Roman', 18),
            bg='#554545',
            fg='#fff'
        )
        headline.place(relx=0.5, rely=0.1, anchor='center')

        # 模型选择下拉菜单
        self.model_choice = tk.StringVar()  # 创建一个Tkinter字符串变量
        model_combobox = ttk.Combobox(
            self.frame,
            textvariable=self.model_choice,
            values=['HubModel', 'PixSynthModel'],
            state='readonly'  # 设置为只读，用户不能自己输入
        )
        model_combobox.place(relx=0.82, rely=0.12, anchor='center')
        model_combobox.current(0)

        # 原图显示框
        self.original_image_frame = tk.Label(
            self.frame,
            image=self.transparent_image
        )
        self.original_image_frame.place(relx=0.2, rely=0.45, anchor='center')

        # 原图打开按钮
        open_original_image = tk.Button(
            self.frame,
            text='Open Original Image',
            font=('Times New Roman', 12),
            bg='#DDDDDD',
            fg='#333333',
            width=18,
            height=1,
            command=lambda: self.browse_img_and_display(self.original_image_frame, is_content=True)
        )
        open_original_image.place(relx=0.2, rely=0.75, anchor='center')

        # 原图标签
        original_image_label = tk.Label(
            self.frame,
            text='Original Image',
            font=('Times New Roman', 15),
            bg='#554545',
            fg='#fff'
        )
        original_image_label.place(relx=0.2, rely=0.85, anchor='center')

        # 风格图显示框
        self.style_image_frame = tk.Label(
            self.frame,
            image=self.transparent_image
        )
        self.style_image_frame.place(relx=0.5, rely=0.45, anchor='center')

        # 风格图打开按钮
        open_style_image = tk.Button(
            self.frame,
            text='Open Style Image',
            font=('Times New Roman', 12),
            bg='#DDDDDD',
            fg='#333333',
            width=18,
            height=1,
            command=lambda: self.browse_img_and_display(self.style_image_frame, is_content=False)
        )
        open_style_image.place(relx=0.5, rely=0.75, anchor='center')

        # 风格图标签
        style_image_label = tk.Label(
            self.frame,
            text='Style Image',
            font=('Times New Roman', 15),
            bg='#554545',
            fg='#fff'
        )
        style_image_label.place(relx=0.5, rely=0.85, anchor='center')

        # 结果图显示框
        self.result_image_frame = tk.Label(
            self.frame,
            image=self.transparent_image
        )
        self.result_image_frame.place(relx=0.8, rely=0.45, anchor='center')

        # 结果图生成按钮
        start_button = tk.Button(
            self.frame,
            text='Start Processing',
            font=('Times New Roman', 12),
            bg='#DDDDDD',
            fg='#333333',
            width=18,
            height=1,
            command=self.process_image_based_on_model_choice
        )
        start_button.place(relx=0.8, rely=0.75, anchor='center')

        # 结果图标签
        result_image_label = tk.Label(
            self.frame,
            text='Result Image',
            font=('Times New Roman', 15),
            bg='#554545',
            fg='#fff'
        )
        result_image_label.place(relx=0.8, rely=0.85, anchor='center')

    def browse_img_and_display(self, image_frame, is_content=True):
        """浏览图片并显示在指定的frame上，同时保存图片路径"""
        file_path = filedialog.askopenfilename()
        if file_path:
            # 保存图片路径
            if is_content:
                self.content_image_path = file_path
            else:
                self.style_image_path = file_path

            img = Image.open(file_path)
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            image_frame.config(image=img_tk)
            image_frame.image = img_tk

    def process_image_based_on_model_choice(self):
        """根据模型选择处理图像"""
        if self.model_choice.get() == 'HubModel':
            self.use_model_hub()
        elif self.model_choice.get() == 'PixSynthModel':
            self.use_model_pixsynth()

    def use_model_hub(self):
        hubModel = HubModel()

        # 风格转换并保存
        content_is_local = True
        style_is_local = True
        save_path = 'result/hub_model/result.png'
        hubModel.style_conver(save_path, self.content_image_path, self.style_image_path, content_is_local,
                              style_is_local)

        img = Image.open(save_path)
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.result_image_frame.config(image=img_tk)
        self.result_image_frame.image = img_tk

    def use_model_pixsynth(self):
        pixsynthModel = PixSynthModel()

        # 风格转换并保存
        content_is_local = True
        style_is_local = True
        save_path = 'result/pixsynth_model/result.png'
        pixsynthModel.style_conver(save_path, self.content_image_path, self.style_image_path, content_is_local,
                                   style_is_local)

        img = Image.open(save_path)
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.result_image_frame.config(image=img_tk)
        self.result_image_frame.image = img_tk


if __name__ == '__main__':
    root = tk.Tk()
    Home(root)
    root.mainloop()
