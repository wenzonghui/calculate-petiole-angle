import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from tools import cat_sub_image, crop_images_in_batch2, detect_deg, find_red_centers, mark_centers, select_rois_from_first_image


# 裁切植株
def jietu(base_path, x_coords, progress_bar=None, progress_label=None):
    for folder in ["pictures"]:
        folder_path = os.path.join(base_path, folder)

        if not os.path.exists(folder_path):
            print(f"文件夹 {folder_path} 不存在")
            continue

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)

        for idx, filename in enumerate(image_files):
            image_path = os.path.join(folder_path, filename)
            cat_sub_image(image_path, x_coords, folder_path)

            # 更新进度条 + 百分比
            if progress_bar:
                percent = (idx + 1) / total_images * 100
                progress_bar["value"] = percent
                if progress_label:
                    progress_label.config(text=f"{int(percent)}%")
                progress_bar.update_idletasks()


# 计算角度
def get_deg(base_path, progress_bar=None, progress_label=None):
    total_processed_images = 0
    total_images = 0

    for folder in ["cropped"]:
        folder_path = os.path.join(base_path, folder)

        if not os.path.exists(folder_path):
            print(f"文件夹 {folder_path} 不存在")
            continue

        sub_folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

        # 先统计总图片数
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(folder_path, sub_folder)
            image_files = [f for f in os.listdir(sub_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(image_files)

        # 再开始处理并更新进度条
        for sub_folder in sorted(sub_folders, key=int):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            angles_output_path = os.path.join(sub_folder_path, "angles.txt")
            A = None
            B = None
            C = None

            with open(angles_output_path, 'w') as file:
                image_files = [f for f in os.listdir(sub_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for img_idx, filename in enumerate(image_files):
                    image_path = os.path.join(sub_folder_path, filename)
                    image = Image.open(image_path)
                    centers = find_red_centers(image)
                    mark_centers(image, centers)

                    red_marked_path = os.path.join(os.path.dirname(image_path), f"red_marked_{os.path.basename(image_path)}")
                    image.save(red_marked_path)

                    if A is None or B is None or C is None:
                        angle, A, B, C = detect_deg(filename, centers)
                    else:
                        angle, A, B, C = detect_deg(filename, centers, A, B)

                    if angle is not None:
                        file.write(f"{filename} {angle:.2f}\n")
                    else:
                        file.write(f"{filename}: 无法计算角度\n")

                    # 每处理一张图就更新进度条
                    total_processed_images += 1
                    if progress_bar and total_images > 0:
                        percent = (total_processed_images / total_images) * 100
                        progress_bar["value"] = percent
                        if progress_label:
                            progress_label.config(text=f"{int(percent)}%")
                        progress_bar.update_idletasks()


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(padx=20, pady=20)
        self.create_widgets()

    def create_widgets(self):
        # 按钮通用样式
        style = {
            "bg": "#4CAF50",
            "fg": "white",
            "font": ("Helvetica", 12, "bold"),
            "relief": tk.FLAT,
            "bd": 0
        }

        label_style = {
            "bg": "#f5f5f5",
            "fg": "#333",
            "font": ("Helvetica", 12)
        }

        # 创建内容容器 frame
        content_frame = tk.Frame(self, bg="#f5f5f5")
        content_frame.grid(row=0, column=0)

        # 控件创建
        self.select_button = tk.Button(content_frame, text="选择图片路径", command=self.select_base_path, **style)
        self.base_path_label = tk.Label(content_frame, text="未选择图片路径", width=40, anchor="w", **label_style)
        self.run_button = tk.Button(content_frame, text="开始计算叶柄夹角", command=self.run_logic, **style)
        self.continue_button = tk.Button(content_frame, text="继续计算", command=self.continue_processing, **style)
        self.quit_button = tk.Button(content_frame, text="退出程序", command=self.master.destroy, fg="white", bg="#f44336",
                                     font=("Helvetica", 12, "bold"), relief=tk.FLAT, bd=0)

        # 进度条容器（带百分比）
        crop_progress_frame = tk.Frame(content_frame, bg="#f5f5f5")
        crop_progress_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)

        angle_progress_frame = tk.Frame(content_frame, bg="#f5f5f5")
        angle_progress_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)

        # 创建带百分比的进度条组件
        self.crop_progress_bar = ttk.Progressbar(crop_progress_frame, orient="horizontal", length=600, mode="determinate")
        self.crop_progress_label = tk.Label(crop_progress_frame, text="0%", width=4, anchor="w", bg="#f5f5f5", font=("Helvetica", 10))

        self.angle_progress_bar = ttk.Progressbar(angle_progress_frame, orient="horizontal", length=600, mode="determinate")
        self.angle_progress_label = tk.Label(angle_progress_frame, text="0%", width=4, anchor="w", bg="#f5f5f5", font=("Helvetica", 10))

        # 布局：水平排列进度条和 Label
        self.crop_progress_bar.pack(side="left")
        self.crop_progress_label.pack(side="left", padx=(5, 0))

        self.angle_progress_bar.pack(side="left")
        self.angle_progress_label.pack(side="left", padx=(5, 0))

        # 添加进度条标签
        self.crop_label = tk.Label(content_frame, text="植株叶柄裁剪", bg="#f5f5f5", font=("Helvetica", 12))
        self.angle_label = tk.Label(content_frame, text="叶柄夹角计算", bg="#f5f5f5", font=("Helvetica", 12))

        # 控件布局
        self.select_button.grid(row=0, column=0, sticky="w", pady=5)
        self.base_path_label.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        self.run_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        self.crop_label.grid(row=2, column=0, sticky="w", pady=(10, 0))
        crop_progress_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)

        self.angle_label.grid(row=4, column=0, sticky="w", pady=(10, 0))
        angle_progress_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)

        self.continue_button.grid(row=6, column=0, sticky="ew", pady=5)
        self.quit_button.grid(row=6, column=1, sticky="ew", pady=5, padx=(5, 0))  # 左侧加点间隔

        # 设置按钮宽度一致
        self.select_button.config(width=20)
        self.run_button.config(width=20)

        # 添加鼠标悬停效果
        def on_enter(e): e.widget['background'] = '#45a049'
        def on_leave(e): e.widget['background'] = '#4CAF50'
        def on_quit_enter(e): e.widget['background'] = '#e53935'
        def on_quit_leave(e): e.widget['background'] = '#f44336'

        self.select_button.bind("<Enter>", on_enter)
        self.select_button.bind("<Leave>", on_leave)

        self.run_button.bind("<Enter>", on_enter)
        self.run_button.bind("<Leave>", on_leave)

        self.quit_button.bind("<Enter>", on_quit_enter)
        self.quit_button.bind("<Leave>", on_quit_leave)

    def select_base_path(self):
        path = filedialog.askdirectory()
        if path:
            self.base_path_label.config(text=path)
            self.base_path = path

    def continue_processing(self):
        print("\n\n###############################################")
        print("用户点击了：继续计算（重新选择路径）")

        # 清除已选路径
        self.base_path_label.config(text="未选择图片路径")
        if hasattr(self, 'base_path'):
            del self.base_path

        # 重置进度条
        self.crop_progress_bar["value"] = 0
        self.angle_progress_bar["value"] = 0
        self.crop_progress_label.config(text="0%")
        self.angle_progress_label.config(text="0%")

    def run_logic(self):
        if hasattr(self, 'base_path'):
            base_path = self.base_path

            # 1、裁切植株
            image_list = sorted([f for f in os.listdir(base_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if image_list:
                first_image_path = os.path.join(base_path, image_list[0])
                rois = select_rois_from_first_image(first_image_path)
            else:
                rois = None
                print("没有找到图片。")

            if rois:
                self.crop_progress_bar["value"] = 0
                self.crop_progress_label.config(text="0%")
                crop_images_in_batch2(base_path, rois, self.crop_progress_bar, self.crop_progress_label)  # 支持传入 label
                print("植株叶柄裁剪处理完成。")
            else:
                print("未选择任何区域，程序退出。")

            # 2、进行角度计算
            self.angle_progress_bar["value"] = 0
            self.angle_progress_label.config(text="0%")
            get_deg(base_path, self.angle_progress_bar, self.angle_progress_label)  # 支持传入 label
            print("叶柄夹角计算完成。")
        else:
            print("请先选择一个基础路径！")


root = tk.Tk()
root.option_add("*Font", "Helvetica 12")  # 设置全局字体为 Helvetica，字号12
root.title("叶柄夹角计算工具")
root.geometry("700x300")      # 设置默认窗口大小
root.minsize(700, 300)        # 设置最小窗口大小
app = Application(master=root)
app.mainloop()