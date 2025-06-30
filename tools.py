import os
from PIL import Image, ImageDraw
import numpy as np
import math
import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk

# 判断一个像素是否是红色的函数
def is_red(pixel, threshold=180):
    r, g, b = pixel
    return r > threshold and g < threshold / 2 and b < threshold / 2

# 计算两点之间的距离
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# 计算红色区域的中心
def calculate_center(red_pixels):
    if len(red_pixels) == 0:
        return None

    # 找到最远的两个点
    max_distance = 0
    point1, point2 = None, None

    for i in range(len(red_pixels)):
        for j in range(i + 1, len(red_pixels)):
            d = distance(red_pixels[i], red_pixels[j])
            if d > max_distance:
                max_distance = d
                point1, point2 = red_pixels[i], red_pixels[j]

    # 计算中心点
    if point1 and point2:
        center_x = (point1[0] + point2[0]) / 2
        center_y = (point1[1] + point2[1]) / 2
        return (center_x, center_y)
    return None

# 读取图像并查找红色区域中心
def find_red_centers(image):
    pixels = np.array(image)

    # 获取图像尺寸
    width, height = image.size

    visited = np.zeros((height, width), dtype=bool)
    centers = []

    # 遍历图像
    for y in range(height):
        for x in range(width):
            if not visited[y, x] and is_red(pixels[y, x]):
                # 找到一个红色区域，进行区域生长以找到完整区域
                red_pixels = []
                stack = [(y, x)]

                while stack:
                    cy, cx = stack.pop()
                    if visited[cy, cx] or not is_red(pixels[cy, cx]):
                        continue

                    visited[cy, cx] = True
                    red_pixels.append((cx, cy))

                    # 检查周围的像素
                    for ny, nx in [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]:
                        if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                            stack.append((ny, nx))

                # 计算红色区域的中心
                center = calculate_center(red_pixels)
                if center:
                    centers.append(center)

    return centers

# 在图像上标记红色区域的中心并保存
def mark_centers(image, centers):
    draw = ImageDraw.Draw(image)
    for center in centers:
        x, y = int(center[0]), int(center[1])
        # 画一个黑色的叉号
        draw.line((x - 10, y - 10, x + 10, y + 10), fill="black", width=4)
        draw.line((x - 10, y + 10, x + 10, y - 10), fill="black", width=4)

# 画线段连接ABC三点
def draw_lines(image, A, B, C):
    draw = ImageDraw.Draw(image)
    draw.line((A, B), fill="black", width=2)
    draw.line((B, C), fill="black", width=2)
    # draw.line((A, C), fill="black", width=2)
    return image

# 按照图中植物的数量截图
def cat_sub_image(image_path, x_coords, output_folder):
    image = Image.open(image_path)
    image_name = os.path.basename(image_path)

    for i, x in enumerate(x_coords):
        if i < len(x_coords) - 1:
            # 截取图片
            box = (x, 0, x_coords[i+1], image.height)
            sub_image = image.crop(box)
            sub_folder = os.path.join(output_folder, str(i + 1))
            os.makedirs(sub_folder, exist_ok=True)
            sub_image.save(os.path.join(sub_folder, image_name))

# 计算向量 BA 和 BC 之间夹角的函数
def calculate_angle(A, B, C):
    # 定义向量 BA 和 BC
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])

    # 计算向量的点积
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]

    # 计算向量的模
    magnitude_BA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
    magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

    # 计算余弦值
    if magnitude_BA * magnitude_BC == 0:
        return 0

    cos_angle = dot_product / (magnitude_BA * magnitude_BC)

    # 使用反余弦计算弧度角
    angle_rad = math.acos(min(1, max(-1, cos_angle)))  # 限制余弦值在 [-1, 1] 范围内

    # 将弧度转换为度数
    angle_deg = math.degrees(angle_rad)

    return angle_deg

# 检测角度
def detect_deg(filename, centers, A=None, B=None):
    if len(centers) == 3:
        # 找到 x 坐标差值最小的两个点
        # 先将3个点按照x坐标进行排序
        sorted_centers = sorted(centers, key=lambda p: p[0])

        x_distance1 = abs(sorted_centers[0][0] - sorted_centers[1][0])
        x_distance2 = abs(sorted_centers[1][0] - sorted_centers[2][0])
        x_distance3 = abs(sorted_centers[0][0] - sorted_centers[2][0])

        if min(x_distance1, x_distance2, x_distance3) == x_distance1:
            C = sorted_centers[2]
            if sorted_centers[0][1] > sorted_centers[1][1]:
                A = sorted_centers[1]
                B = sorted_centers[0]
            else:
                A = sorted_centers[0]
                B = sorted_centers[1]
        elif min(x_distance1, x_distance2, x_distance3) == x_distance2:
            C = sorted_centers[0]
            if sorted_centers[1][1] > sorted_centers[2][1]:
                A = sorted_centers[2]
                B = sorted_centers[1]
            else:
                A = sorted_centers[2]
                B = sorted_centers[1]
        elif min(x_distance1, x_distance2, x_distance3) == x_distance3:
            C = sorted_centers[1]
            if sorted_centers[0][1] > sorted_centers[2][1]:
                A = sorted_centers[2]
                B = sorted_centers[0]
            else:
                A = sorted_centers[0]
                B = sorted_centers[2]

        angle = calculate_angle(A, B, C)
        print(f"叶柄夹角计算：{filename}: {angle:.2f} degrees")
        return angle, A, B, C
    elif len(centers) == 1:
        C = centers[0]
        angle = calculate_angle(A, B, C)
        print(f"叶柄夹角计算：{filename}: {angle:.2f} degrees")
        return angle, A, B, C
    else:
        print(f"在图片 {os.path.basename(filename)} 中未检测到红色中心点: {len(centers)}")
        return None, None, None, None


class RoiSelectorDialog:
    def __init__(self, master, image_path):
        self.master = master
        self.image_path = image_path
        self.rois = []

        # 加载图像
        self.original_img = Image.open(image_path)
        self.display_img = self.original_img.copy()

        # 设置最大宽度为 1080，保持宽高比
        MAX_WIDTH = 1080
        scale_factor = MAX_WIDTH / self.original_img.width
        new_height = int(self.original_img.height * scale_factor)

        self.display_img = self.original_img.resize(
            (MAX_WIDTH, new_height),
            Image.Resampling.LANCZOS
        )

        self.tk_img = ImageTk.PhotoImage(self.display_img)

        # 创建新窗口，尺寸自适应图像
        self.top = tk.Toplevel(master)
        self.top.title("选择截图区域")
        self.top.geometry(f"{MAX_WIDTH}x{new_height + 80}")  # 预留按钮空间

        # Canvas 显示图像
        self.canvas = tk.Canvas(self.top, width=MAX_WIDTH, height=new_height, highlightthickness=0)
        self.canvas.pack()

        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # 存储所有框和组件
        self.current_rects = []  # (x1, y1, x2, y2, rect_id, label_id, delete_id)

        # 事件绑定
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # 按钮
        btn_frame = tk.Frame(self.top)
        btn_frame.pack(pady=10)

        # 按钮通用样式（可提取为变量复用）
        btn_style = {
            "bg": "#4CAF50",
            "fg": "white",
            "font": ("Helvetica", 14, "bold"),  # 调大字体至 14 号
            "relief": tk.FLAT,
            "bd": 0,
            "width": 12  # 固定宽度便于对齐
        }

        # 创建按钮
        self.confirm_btn = tk.Button(btn_frame, text="确认", command=self.on_confirm, **btn_style)
        self.confirm_btn.pack(side="left", padx=5)

        self.cancel_btn = tk.Button(btn_frame, text="取消", command=self.on_cancel, **btn_style)
        self.cancel_btn.pack(side="left", padx=5)

        def on_enter(e): e.widget['background'] = '#45a049'
        def on_leave(e): e.widget['background'] = '#4CAF50'

        self.confirm_btn.bind("<Enter>", on_enter)
        self.confirm_btn.bind("<Leave>", on_leave)

        self.cancel_btn.bind("<Enter>", on_enter)
        self.cancel_btn.bind("<Leave>", on_leave)

    def on_mouse_down(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="blue", width=2
        )

    def on_mouse_move(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_mouse_up(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            # 绘制矩形（蓝色）
            rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2)

            # 编号文字（左上角）
            label_id = self.canvas.create_text(x1+1, y1 - 25, text=f"{len(self.current_rects)+1}",
                                              fill="blue", anchor="nw", font=("Arial", 17, "bold"))

            # 叉号删除图标（右上角）
            delete_x = x2
            delete_y = y1 - 20
            delete_id = self.canvas.create_text(delete_x, delete_y, text="❌",
                                                fill="blue", anchor="ne", font=("Arial", 12, "bold"),
                                                tags="delete")

            # 绑定点击事件
            self.canvas.tag_bind(delete_id, "<Button-1>",
                                 lambda e, idx=len(self.current_rects): self.delete_rect(idx))

            # 保存
            self.current_rects.append((x1, y1, x2, y2, rect_id, label_id, delete_id))
            print(f"添加框选区域 {len(self.current_rects)}: ({x1}, {y1}) - ({x2}, {y2})")

        self.canvas.delete(self.rect_id)
        self.rect_id = None

    def delete_rect(self, index):
        if 0 <= index < len(self.current_rects):
            x1, y1, x2, y2, rect_id, label_id, delete_id = self.current_rects[index]
            self.canvas.delete(rect_id)
            self.canvas.delete(label_id)
            self.canvas.delete(delete_id)
            del self.current_rects[index]
            self.redraw_labels(index)

    def redraw_labels(self, start_index=0):
        for i, (x1, y1, x2, y2, rect_id, label_id, delete_id) in enumerate(self.current_rects):
            if i >= start_index:
                # 更新编号文本
                self.canvas.itemconfig(label_id, text=f"{i+1}")
                # 移动编号到左上角
                self.canvas.coords(label_id, x1 + 1, y1 - 25)

            # 移动叉号到右上角
            self.canvas.coords(delete_id, x2, y1 - 20)

            # 更新叉号点击事件绑定
            self.canvas.tag_unbind(delete_id, "<Button-1>")
            self.canvas.tag_bind(delete_id, "<Button-1>",
                                 lambda e, idx=i: self.delete_rect(idx))

    def on_confirm(self):
        scale_w = self.original_img.width / self.display_img.width
        scale_h = self.original_img.height / self.display_img.height
        rois = [
            (int(x1 * scale_w), int(y1 * scale_h), int((x2 - x1) * scale_w), int((y2 - y1) * scale_h))
            for (x1, y1, x2, y2, _, _, _) in self.current_rects
        ]
        self.top.destroy()
        self.result = rois

    def on_cancel(self):
        self.top.destroy()
        self.result = []


def select_rois_from_first_image(first_image_path):
    print("打开 Tkinter ROI 框选窗口...")
    dialog = RoiSelectorDialog(None, first_image_path)
    dialog.top.grab_set()  # 模态对话框
    dialog.top.wait_window()  # 等待用户操作完成
    return getattr(dialog, "result", [])



# 批量截图
def crop_images_in_batch(image_dir, rois):
    save_base = os.path.join(image_dir, "cropped")
    os.makedirs(save_base, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 创建 crop_x 子文件夹
    for idx in range(len(rois)):
        os.makedirs(os.path.join(save_base, f"{idx+1}"), exist_ok=True)

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        for idx, roi in enumerate(rois):
            x, y, w, h = roi
            crop = img[y:y+h, x:x+w]
            save_path = os.path.join(save_base, f"{idx+1}", img_file)
            cv2.imwrite(save_path, crop)
        print(f"进行植株叶柄裁剪：{img_file}")


def crop_images_in_batch2(image_dir, rois, progress_bar=None, progress_label=None):
    save_base = os.path.join(image_dir, "cropped")
    os.makedirs(save_base, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    total_images = len(image_files)

    # 创建子文件夹
    for idx in range(len(rois)):
        os.makedirs(os.path.join(save_base, f"{idx + 1}"), exist_ok=True)

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        for roi_idx, roi in enumerate(rois):
            x, y, w, h = roi
            crop = img[y:y+h, x:x+w]
            save_path = os.path.join(save_base, f"{roi_idx + 1}", img_file)
            cv2.imwrite(save_path, crop)

        # 更新进度条和百分比标签
        if progress_bar:
            percent = (idx + 1) / total_images * 100
            progress_bar["value"] = percent
            if progress_label:
                progress_label.config(text=f"{int(percent)}%")
            progress_bar.update_idletasks()

        print(f"进行植株叶柄裁剪：{img_file}")