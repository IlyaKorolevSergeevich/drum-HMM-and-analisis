import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import multiprocessing as mp
import queue
import time
import pygame
import sys
import os
from screeninfo import get_monitors
import cv2
import datetime
from PIL import Image, ImageTk
import numpy as np
import threading
import math
import random
import screen_brightness_control as sbc
import json

# ==================== КОНСТАНТЫ ====================
MIN_SPATIAL_FREQ = 0.01
MAX_SPATIAL_FREQ = 1.0
DEFAULT_VIEWING_DISTANCE_CM = 30.0
DEFAULT_SCREEN_DIAGONAL_INCH = 24
DEFAULT_SCREEN_WIDTH_PX = 1920
DEFAULT_SCREEN_HEIGHT_PX = 1080

# ==================== УТИЛИТЫ ====================
def compute_screen_angle_deg(distance_cm, screen_diagonal_inch, screen_width_px, screen_height_px):
    diagonal_cm = screen_diagonal_inch * 2.54
    aspect_ratio = screen_width_px / screen_height_px
    screen_width_cm = diagonal_cm * screen_width_px / math.sqrt(screen_width_px**2 + screen_height_px**2)
    angle_rad = 2 * math.atan((screen_width_cm / 2) / distance_cm)
    return math.degrees(angle_rad)

def spatial_freq_to_period_px(spatial_freq_cyc_deg, screen_width_deg, screen_width_px):
    if spatial_freq_cyc_deg <= 0:
        return 50
    deg_per_cycle = 1.0 / spatial_freq_cyc_deg
    period_px = (deg_per_cycle / screen_width_deg) * screen_width_px
    return max(2, int(period_px))

def generate_grating_frame(width, height, period_px, phase, contrast, gamma, color_rgb, pattern_type='sinusoidal'):
    if period_px < 2:
        period_px = 2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        norm_x = ((x - phase) % period_px) / period_px
        if pattern_type == 'sinusoidal':
            intensity = 0.5 + 0.5 * math.sin(2 * math.pi * norm_x)
        elif pattern_type == 'square':
            intensity = 1.0 if norm_x < 0.5 else 0.0
        elif pattern_type == 'sawtooth':
            intensity = norm_x
        elif pattern_type == 'random':
            intensity = random.random()
        else:
            intensity = 0.5 + 0.5 * math.sin(2 * math.pi * norm_x)
        intensity = 0.5 + (intensity - 0.5) * contrast
        corrected = intensity ** (1.0 / gamma)
        val = int(corrected * 255)
        r = int(color_rgb[0] * val / 255)
        g = int(color_rgb[1] * val / 255)
        b = int(color_rgb[2] * val / 255)
        img[:, x, :] = (r, g, b)
    return img

# ==================== ПРОЦЕСС ОТОБРАЖЕНИЯ ====================
def display_process(command_queue, monitor_index, screen_params_queue):
    pygame.init()
    monitors = get_monitors()
    m = monitors[monitor_index]
    width, height = m.width, m.height
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{m.x},{m.y}"
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Стимул")
    screen_width_deg = 100.0
    try:
        screen_params = screen_params_queue.get(timeout=2)
        screen_width_deg = screen_params.get('screen_width_deg', 100.0)
    except:
        pass

    stimulus_active = False
    phase = 0.0
    last_time = time.time()
    current_params = {
        'period_px': 50,
        'contrast': 1.0,
        'gamma': 2.2,
        'color_rgb': (255,255,255),
        'velocity_deg_sec': 12.0,
        'direction': 1,
        'pattern_type': 'sinusoidal'
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        try:
            cmd = command_queue.get_nowait()
            if cmd[0] == "START":
                params = cmd[1]
                spatial_freq = params.get('spatial_freq_cyc_deg', 0.2)
                period_px = spatial_freq_to_period_px(spatial_freq, screen_width_deg, width)
                current_params.update({
                    'period_px': period_px,
                    'contrast': params.get('contrast', 1.0),
                    'gamma': params.get('gamma', 2.2),
                    'color_rgb': params.get('color_rgb', (255,255,255)),
                    'velocity_deg_sec': params.get('velocity_deg_sec', 12.0),
                    'direction': params.get('direction', 1),
                    'pattern_type': params.get('pattern_type', 'sinusoidal')
                })
                stimulus_active = True
                phase = 0.0
                last_time = time.time()
            elif cmd[0] == "STOP":
                stimulus_active = False
                screen.fill((0,0,0))
                pygame.display.flip()
            elif cmd[0] == "QUIT":
                running = False
        except queue.Empty:
            pass

        if stimulus_active:
            now = time.time()
            dt = min(now - last_time, 0.1)
            last_time = now
            vel_px_sec = (current_params['velocity_deg_sec'] / screen_width_deg) * width
            phase += current_params['direction'] * vel_px_sec * dt
            phase %= current_params['period_px']
            img_array = generate_grating_frame(
                width, height,
                current_params['period_px'],
                phase,
                current_params['contrast'],
                current_params['gamma'],
                current_params['color_rgb'],
                current_params['pattern_type']
            )
            surf = pygame.surfarray.make_surface(img_array.swapaxes(0,1))
            screen.blit(surf, (0,0))
            pygame.display.flip()

        pygame.time.wait(10)

    pygame.quit()
    sys.exit(0)

# ==================== ПРОЦЕСС КАМЕРЫ ====================
def camera_process(command_queue, video_queue, camera_id=0):
    cap = None
    for i in range(camera_id, camera_id+5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
        cap.release()
        cap = None
    fake_camera = cap is None
    if fake_camera:
        print("[Camera] Режим эмуляции")
    else:
        print(f"[Camera] Камера {camera_id} открыта")

    recording = False
    writer = None
    running = True
    while running:
        try:
            cmd = command_queue.get_nowait()
            if cmd[0] == "START_RECORDING":
                filepath = cmd[1]
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(filepath, fourcc, 30, (640,480))
                recording = True
                try:
                    video_queue.put_nowait(("FILENAME", os.path.basename(filepath)))
                except queue.Full:
                    pass
            elif cmd[0] == "STOP_RECORDING":
                if writer:
                    writer.release()
                writer = None
                recording = False
                try:
                    video_queue.put_nowait(("STOP",))
                except queue.Full:
                    pass
            elif cmd[0] == "QUIT":
                running = False
        except queue.Empty:
            pass

        if fake_camera:
            frame = np.zeros((480,640,3), dtype=np.uint8)
            cv2.putText(frame, "Camera Emulation", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
        if recording and writer and not fake_camera:
            writer.write(frame)
        # Отправляем каждый кадр для предпросмотра
        small = cv2.resize(frame, (640,480))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        h,w,_ = small_rgb.shape
        try:
            video_queue.put_nowait(("FRAME", small_rgb.tobytes(), w, h))
        except queue.Full:
            pass
        time.sleep(0.01)

    if writer:
        writer.release()
    if cap:
        cap.release()
    sys.exit(0)

# ==================== ГЛАВНОЕ ОКНО ====================
class ModernMainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OMR-Arena | Stimulation System")
        self.root.geometry("1500x900")
        self.root.configure(bg="#0f1219")
        self.setup_styles()

        # Параметры экрана
        self.screen_diagonal = tk.DoubleVar(value=DEFAULT_SCREEN_DIAGONAL_INCH)
        self.viewing_distance = tk.DoubleVar(value=DEFAULT_VIEWING_DISTANCE_CM)
        self.screen_width_px = tk.IntVar(value=DEFAULT_SCREEN_WIDTH_PX)
        self.screen_height_px = tk.IntVar(value=DEFAULT_SCREEN_HEIGHT_PX)
        self.screen_width_deg = 100.0

        # Текущие параметры стимула
        self.pattern_type = tk.StringVar(value="sinusoidal")
        self.spatial_freq = tk.DoubleVar(value=0.2)
        self.contrast = tk.IntVar(value=100)
        self.gamma = tk.DoubleVar(value=2.2)
        self.velocity = tk.DoubleVar(value=12.0)
        self.direction = tk.IntVar(value=1)
        self.duration = tk.DoubleVar(value=5.0)
        self.color_rgb = (255, 255, 255)
        self.stimulus_brightness = tk.IntVar(value=100)

        # Список элементов серии
        self.tests = []

        # Очереди и процессы
        self.display_queue = mp.Queue()
        self.camera_queue = mp.Queue()
        self.video_queue = mp.Queue()
        self.screen_params_queue = mp.Queue()

        # Запуск процессов
        monitors = get_monitors()
        stim_monitor = 0 if len(monitors) < 2 else 0
        self.display_proc = mp.Process(target=display_process, args=(self.display_queue, stim_monitor, self.screen_params_queue))
        self.display_proc.start()
        self.camera_proc = mp.Process(target=camera_process, args=(self.camera_queue, self.video_queue, 0))
        self.camera_proc.start()

        self.update_screen_geometry()
        self.screen_params_queue.put({'screen_width_deg': self.screen_width_deg})

        self.create_widgets()
        self.root.after(100, self.process_video)
        self.root.after(50, self.update_preview_loop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._preview_phase = 0.0
        self.original_brightness = None
        self.paused = False
        self.pending_restart_item = None   # параметры стимула, который нужно повторить после паузы

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#0f1219')
        style.configure('TLabel', background='#0f1219', foreground='#d4d8e8', font=('Inter', 10))
        style.configure('TEntry', fieldbackground='#1e2436', foreground='#ffffff', insertcolor='#ff5e7c', borderwidth=0)
        style.configure('TScale', background='#0f1219', troughcolor='#2a3147', slidercolor='#ff5e7c')
        style.configure('TLabelframe', background='#0f1219', foreground='#ff5e7c', bordercolor='#2a3147')
        style.configure('TLabelframe.Label', background='#0f1219', foreground='#ff5e7c', font=('Inter', 10, 'bold'))
        style.configure('Dark.TLabelframe', background='#0f1219', bordercolor='#2a3147', relief='solid', borderwidth=1)
        style.configure('Dark.TLabelframe.Label', background='#0f1219', foreground='#ff5e7c', font=('Inter', 10, 'bold'))
        style.configure('Dark.TFrame', background='#0f1219')
        style.configure('Dark.TRadiobutton',
                        background='#1e2436',
                        foreground='#ffffff',
                        focuscolor='#ff5e7c',
                        selectcolor='#ff5e7c',
                        font=('Inter', 9))
        style.map('Dark.TRadiobutton',
                  background=[('active', '#ff5e7c'), ('selected', '#ff5e7c')],
                  foreground=[('active', '#ffffff'), ('selected', '#ffffff')])
        style.configure('Dark.TButton',
                        background='#252b3d',
                        foreground='#ffffff',
                        borderwidth=0,
                        font=('Inter', 9, 'bold'))
        style.map('Dark.TButton',
                  background=[('active', '#ff5e7c')],
                  foreground=[('active', '#ffffff')])

    def update_screen_geometry(self):
        self.screen_width_deg = compute_screen_angle_deg(
            self.viewing_distance.get(),
            self.screen_diagonal.get(),
            self.screen_width_px.get(),
            self.screen_height_px.get()
        )
        self.screen_params_queue.put({'screen_width_deg': self.screen_width_deg})

    def set_brightness(self, brightness_percent):
        try:
            sbc.set_brightness(brightness_percent)
        except Exception as e:
            print(f"Не удалось изменить яркость: {e}")

    def create_widgets(self):
        # Верхняя панель с геометрией
        geo_frame = ttk.LabelFrame(self.root, text="Монитор и геометрия", padding=8)
        geo_frame.pack(fill='x', padx=15, pady=(10,5))
        for i, (label, var) in enumerate([
            ("Диагональ (дюймы):", self.screen_diagonal),
            ("Расстояние (см):", self.viewing_distance),
            ("Ширина (px):", self.screen_width_px),
            ("Высота (px):", self.screen_height_px)
        ]):
            ttk.Label(geo_frame, text=label).grid(row=0, column=i*2, padx=5)
            entry = ttk.Entry(geo_frame, textvariable=var, width=8)
            entry.grid(row=0, column=i*2+1, padx=5)
        ttk.Button(geo_frame, text="Обновить", command=self.update_screen_geometry, style='Dark.TButton').grid(row=0, column=8, padx=10)

        # Основной контейнер
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=15, pady=10)

        # Левая панель
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0,10))

        # Предпросмотр стимула
        preview_frame = ttk.LabelFrame(left_panel, text="Предпросмотр стимула (текущие настройки)", padding=5)
        preview_frame.pack(fill='both', expand=True, pady=5)
        self.preview_canvas = tk.Canvas(preview_frame, bg='#000000', highlightthickness=0, relief='flat')
        self.preview_canvas.pack(fill='both', expand=True)

        # Предпросмотр камеры (Canvas)
        cam_frame = ttk.LabelFrame(left_panel, text="Камера", padding=5)
        cam_frame.pack(fill='both', expand=True, pady=5)
        cam_container = ttk.Frame(cam_frame)
        cam_container.pack(fill='both', expand=True)
        cam_container.pack_propagate(False)
        cam_container.config(height=300)
        self.cam_canvas = tk.Canvas(cam_container, bg='#000000', highlightthickness=0)
        self.cam_canvas.pack(fill='both', expand=True)

        # Правая панель с прокруткой
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side='right', fill='both', expand=True)

        canvas = tk.Canvas(right_panel, bg='#0f1219', highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_panel, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw', width=canvas.winfo_width())
        canvas.configure(yscrollcommand=scrollbar.set)

        def _configure_canvas(event):
            canvas.itemconfig("all", width=event.width)
        canvas.bind("<Configure>", _configure_canvas)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ========== БЛОК НАСТРОЕК СТИМУЛА ==========
        settings_frame = ttk.LabelFrame(scrollable_frame, text="Настройки текущего стимула", padding=12, style='Dark.TLabelframe')
        settings_frame.pack(fill='x', pady=5)

        row = 0
        ttk.Label(settings_frame, text="Тип стимула:").grid(row=row, column=0, sticky='w', pady=6)
        pattern_buttons = ttk.Frame(settings_frame, style='Dark.TFrame')
        pattern_buttons.grid(row=row, column=1, columnspan=3, sticky='w', padx=10)
        for text, val in [("Синусоидальный", "sinusoidal"), ("Квадратный", "square"), ("Пилообразный", "sawtooth"), ("Случайные точки", "random")]:
            ttk.Radiobutton(pattern_buttons, text=text, variable=self.pattern_type, value=val, style='Dark.TRadiobutton').pack(side='left', padx=5)
        row += 1

        ttk.Label(settings_frame, text="Пространственная частота (циклы/град):").grid(row=row, column=0, sticky='w', pady=6)
        sf_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        sf_frame.grid(row=row, column=1, columnspan=3, sticky='we', padx=10)
        self.sf_scale = ttk.Scale(sf_frame, from_=MIN_SPATIAL_FREQ, to=MAX_SPATIAL_FREQ, orient='horizontal', variable=self.spatial_freq, length=250)
        self.sf_scale.pack(side='left')
        self.sf_entry = ttk.Entry(sf_frame, textvariable=self.spatial_freq, width=8)
        self.sf_entry.pack(side='left', padx=10)
        row += 1

        ttk.Label(settings_frame, text="Контраст (%):").grid(row=row, column=0, sticky='w', pady=6)
        cont_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        cont_frame.grid(row=row, column=1, columnspan=3, sticky='we', padx=10)
        self.cont_scale = ttk.Scale(cont_frame, from_=0, to=100, orient='horizontal', variable=self.contrast, length=250)
        self.cont_scale.pack(side='left')
        self.cont_entry = ttk.Entry(cont_frame, textvariable=self.contrast, width=8)
        self.cont_entry.pack(side='left', padx=10)
        row += 1

        ttk.Label(settings_frame, text="Гамма:").grid(row=row, column=0, sticky='w', pady=6)
        gamma_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        gamma_frame.grid(row=row, column=1, columnspan=3, sticky='we', padx=10)
        self.gamma_scale = ttk.Scale(gamma_frame, from_=1.0, to=3.0, orient='horizontal', variable=self.gamma, length=250)
        self.gamma_scale.pack(side='left')
        self.gamma_entry = ttk.Entry(gamma_frame, textvariable=self.gamma, width=8)
        self.gamma_entry.pack(side='left', padx=10)
        row += 1

        ttk.Label(settings_frame, text="Скорость (°/с):").grid(row=row, column=0, sticky='w', pady=6)
        vel_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        vel_frame.grid(row=row, column=1, columnspan=3, sticky='we', padx=10)
        self.vel_scale = ttk.Scale(vel_frame, from_=0, to=100, orient='horizontal', variable=self.velocity, length=250)
        self.vel_scale.pack(side='left')
        self.vel_entry = ttk.Entry(vel_frame, textvariable=self.velocity, width=8)
        self.vel_entry.pack(side='left', padx=10)
        row += 1

        ttk.Label(settings_frame, text="Направление:").grid(row=row, column=0, sticky='w', pady=6)
        dir_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        dir_frame.grid(row=row, column=1, columnspan=3, sticky='w', padx=10)
        ttk.Radiobutton(dir_frame, text="Влево ◀", variable=self.direction, value=1, style='Dark.TRadiobutton').pack(side='left', padx=10)
        ttk.Radiobutton(dir_frame, text="Вправо ▶", variable=self.direction, value=-1, style='Dark.TRadiobutton').pack(side='left', padx=10)
        row += 1

        ttk.Label(settings_frame, text="Длительность стимула (с):").grid(row=row, column=0, sticky='w', pady=6)
        dur_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        dur_frame.grid(row=row, column=1, columnspan=3, sticky='we', padx=10)
        ttk.Entry(dur_frame, textvariable=self.duration, width=8).pack(side='left')
        row += 1

        ttk.Label(settings_frame, text="Цвет стимула:").grid(row=row, column=0, sticky='w', pady=6)
        color_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        color_frame.grid(row=row, column=1, columnspan=3, sticky='w', padx=10)
        self.color_preview = tk.Canvas(color_frame, width=30, height=20, bg='#ffffff', highlightthickness=0)
        self.color_preview.pack(side='left')
        ttk.Button(color_frame, text="Выбрать", command=self.choose_color, style='Dark.TButton').pack(side='left', padx=10)
        row += 1

        ttk.Label(settings_frame, text="Яркость стимула (%):").grid(row=row, column=0, sticky='w', pady=6)
        bright_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        bright_frame.grid(row=row, column=1, columnspan=3, sticky='we', padx=10)
        self.bright_scale = ttk.Scale(bright_frame, from_=0, to=100, orient='horizontal', variable=self.stimulus_brightness, length=250)
        self.bright_scale.pack(side='left')
        self.bright_entry = ttk.Entry(bright_frame, textvariable=self.stimulus_brightness, width=8)
        self.bright_entry.pack(side='left', padx=10)
        row += 1

        btn_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        btn_frame.grid(row=row, column=0, columnspan=4, pady=10)
        ttk.Button(btn_frame, text="+ Добавить стимул в список", command=self.add_current_stimulus, style='Dark.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="+ Добавить чёрный экран", command=self.add_black_screen, style='Dark.TButton').pack(side='left', padx=5)
        row += 1

        # Список тестов
        tests_frame = ttk.LabelFrame(scrollable_frame, text="Список тестов (будет выполнен по порядку)", padding=5)
        tests_frame.pack(fill='both', expand=True, pady=5)

        list_buttons = ttk.Frame(tests_frame)
        list_buttons.pack(fill='x', pady=5)
        ttk.Button(list_buttons, text="Редактировать", command=self.edit_test, style='Dark.TButton').pack(side='left', padx=5)
        ttk.Button(list_buttons, text="Удалить", command=self.remove_test, style='Dark.TButton').pack(side='left', padx=5)
        ttk.Button(list_buttons, text="Очистить все", command=self.clear_tests, style='Dark.TButton').pack(side='left', padx=5)

        self.test_listbox = tk.Listbox(tests_frame, bg='#1e2436', fg='#ffffff', selectbackground='#ff5e7c', selectforeground='#ffffff', font=('Consolas', 9))
        self.test_listbox.pack(side='left', fill='both', expand=True)
        scroll_list = ttk.Scrollbar(tests_frame, orient='vertical', command=self.test_listbox.yview)
        scroll_list.pack(side='right', fill='y')
        self.test_listbox.configure(yscrollcommand=scroll_list.set)

        # Панель запуска
        run_frame = ttk.LabelFrame(scrollable_frame, text="Управление экспериментом", padding=8)
        run_frame.pack(fill='x', pady=5)

        ttk.Label(run_frame, text="Задержка перед серией (с):").grid(row=0, column=0, sticky='w', pady=5)
        self.pre_delay_var = tk.DoubleVar(value=2.0)
        ttk.Entry(run_frame, textvariable=self.pre_delay_var, width=8).grid(row=0, column=1, sticky='w', padx=5)

        ttk.Label(run_frame, text="Префикс видео:").grid(row=1, column=0, sticky='w', pady=5)
        self.prefix_var = tk.StringVar()
        ttk.Entry(run_frame, textvariable=self.prefix_var, width=15).grid(row=1, column=1, sticky='w', padx=5)

        ttk.Label(run_frame, text="Папка сохранения:").grid(row=2, column=0, sticky='w', pady=5)
        self.save_dir_var = tk.StringVar(value="recordings")
        ttk.Entry(run_frame, textvariable=self.save_dir_var, width=15).grid(row=2, column=1, sticky='w', padx=5)
        ttk.Button(run_frame, text="Выбрать", command=self.choose_folder, style='Dark.TButton').grid(row=2, column=2, padx=5)

        # Кнопки сохранения/загрузки серии
        save_load_frame = ttk.Frame(run_frame)
        save_load_frame.grid(row=3, column=0, columnspan=3, pady=5)
        ttk.Button(save_load_frame, text="💾 Сохранить серию", command=self.save_series, style='Dark.TButton').pack(side='left', padx=5)
        ttk.Button(save_load_frame, text="📂 Загрузить серию", command=self.load_series, style='Dark.TButton').pack(side='left', padx=5)

        # Кнопка паузы (изначально неактивна)
        self.pause_btn = ttk.Button(run_frame, text="⏸ Пауза", command=self.toggle_pause, style='Dark.TButton', state='disabled')
        self.pause_btn.grid(row=4, column=0, columnspan=1, pady=5, padx=5)

        self.start_btn = ttk.Button(run_frame, text="▶ СТАРТ СЕРИИ", command=self.start_series, style='Dark.TButton')
        self.start_btn.grid(row=4, column=1, columnspan=2, pady=5)

        # Статусная строка
        self.status_var = tk.StringVar(value="✅ Готов к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='flat', anchor='w', background='#0f1219', foreground='#a0a8c0')
        status_bar.pack(side='bottom', fill='x', padx=15, pady=(0,10))

    def choose_color(self):
        color = colorchooser.askcolor(initialcolor='#%02x%02x%02x' % self.color_rgb, title="Цвет стимула")
        if color[0]:
            self.color_rgb = tuple(int(c) for c in color[0])
            self.color_preview.config(bg='#%02x%02x%02x' % self.color_rgb)

    def add_current_stimulus(self):
        test = {
            'type': 'stimulus',
            'pattern_type': self.pattern_type.get(),
            'spatial_freq_cyc_deg': self.spatial_freq.get(),
            'contrast': self.contrast.get() / 100.0,
            'gamma': self.gamma.get(),
            'velocity_deg_sec': self.velocity.get(),
            'direction': self.direction.get(),
            'duration': self.duration.get(),
            'color_rgb': self.color_rgb,
            'brightness': self.stimulus_brightness.get()
        }
        self.tests.append(test)
        self.update_test_listbox()

    def add_black_screen(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Чёрный экран")
        dialog.configure(bg="#0f1219")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(dialog, text="Длительность (с):", background="#0f1219", foreground="white").pack(padx=20, pady=10)
        dur_var = tk.DoubleVar(value=2.0)
        entry = ttk.Entry(dialog, textvariable=dur_var, width=10)
        entry.pack(padx=20, pady=5)
        def confirm():
            self.tests.append({'type': 'black_screen', 'duration': dur_var.get()})
            self.update_test_listbox()
            dialog.destroy()
        ttk.Button(dialog, text="OK", command=confirm, style='Dark.TButton').pack(pady=10)
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def save_series(self):
        if not self.tests:
            messagebox.showwarning("Нет данных", "Нечего сохранять. Добавьте стимулы или чёрные экраны.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filepath:
            try:
                serializable_tests = []
                for t in self.tests:
                    t_copy = t.copy()
                    if 'color_rgb' in t_copy:
                        t_copy['color_rgb'] = list(t_copy['color_rgb'])
                    serializable_tests.append(t_copy)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(serializable_tests, f, indent=2, ensure_ascii=False)
                self.status_var.set(f"Серия сохранена: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить: {e}")

    def load_series(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                self.tests = []
                for t in loaded:
                    if 'color_rgb' in t:
                        t['color_rgb'] = tuple(t['color_rgb'])
                    self.tests.append(t)
                self.update_test_listbox()
                self.status_var.set(f"Серия загружена: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить: {e}")

    def edit_test(self):
        idx = self.test_listbox.curselection()
        if not idx:
            messagebox.showinfo("Нет выбора", "Выберите элемент для редактирования")
            return
        idx = idx[0]
        item = self.tests[idx]
        if item['type'] == 'stimulus':
            self._edit_stimulus(idx, item)
        else:
            self._edit_black_screen(idx, item)

    def _edit_stimulus(self, idx, test):
        dialog = tk.Toplevel(self.root)
        dialog.title("Редактировать стимул")
        dialog.configure(bg="#0f1219")
        dialog.transient(self.root)
        dialog.grab_set()
        pattern_ru = {"sinusoidal":"синусоидальный", "square":"квадратный", "sawtooth":"пилообразный", "random":"случайные точки"}
        pattern_rev = {v: k for k, v in pattern_ru.items()}
        pattern_var = tk.StringVar(value=pattern_ru.get(test['pattern_type'], "синусоидальный"))
        sf = tk.DoubleVar(value=test['spatial_freq_cyc_deg'])
        cont = tk.IntVar(value=int(test['contrast']*100))
        gamma = tk.DoubleVar(value=test['gamma'])
        vel = tk.DoubleVar(value=test['velocity_deg_sec'])
        direc = tk.IntVar(value=test['direction'])
        dur = tk.DoubleVar(value=test['duration'])
        color_rgb = test['color_rgb']
        brightness = tk.IntVar(value=test.get('brightness', 100))

        row = 0
        ttk.Label(dialog, text="Тип:", background="#0f1219", foreground="white").grid(row=row, column=0, padx=10, pady=5, sticky='e')
        pattern_combo = ttk.Combobox(dialog, textvariable=pattern_var, values=list(pattern_ru.values()), state='readonly', width=20)
        pattern_combo.grid(row=row, column=1, padx=10, pady=5, sticky='w')
        row += 1
        ttk.Label(dialog, text="Частота (циклы/град):", background="#0f1219", foreground="white").grid(row=row, column=0, padx=10, pady=5, sticky='e')
        ttk.Entry(dialog, textvariable=sf, width=10).grid(row=row, column=1, padx=10, pady=5, sticky='w')
        row += 1
        ttk.Label(dialog, text="Контраст (%):", background="#0f1219", foreground="white").grid(row=row, column=0, padx=10, pady=5, sticky='e')
        ttk.Entry(dialog, textvariable=cont, width=10).grid(row=row, column=1, padx=10, pady=5, sticky='w')
        row += 1
        ttk.Label(dialog, text="Гамма:", background="#0f1219", foreground="white").grid(row=row, column=0, padx=10, pady=5, sticky='e')
        ttk.Entry(dialog, textvariable=gamma, width=10).grid(row=row, column=1, padx=10, pady=5, sticky='w')
        row += 1
        ttk.Label(dialog, text="Скорость (°/с):", background="#0f1219", foreground="white").grid(row=row, column=0, padx=10, pady=5, sticky='e')
        ttk.Entry(dialog, textvariable=vel, width=10).grid(row=row, column=1, padx=10, pady=5, sticky='w')
        row += 1
        ttk.Label(dialog, text="Направление:", background="#0f1219", foreground="white").grid(row=row, column=0, padx=10, pady=5, sticky='e')
        dir_frame = ttk.Frame(dialog)
        dir_frame.grid(row=row, column=1, padx=10, pady=5, sticky='w')
        ttk.Radiobutton(dir_frame, text="Влево", variable=direc, value=1, style='Dark.TRadiobutton').pack(side='left', padx=5)
        ttk.Radiobutton(dir_frame, text="Вправо", variable=direc, value=-1, style='Dark.TRadiobutton').pack(side='left', padx=5)
        row += 1
        ttk.Label(dialog, text="Длительность (с):", background="#0f1219", foreground="white").grid(row=row, column=0, padx=10, pady=5, sticky='e')
        ttk.Entry(dialog, textvariable=dur, width=10).grid(row=row, column=1, padx=10, pady=5, sticky='w')
        row += 1
        ttk.Label(dialog, text="Яркость (%):", background="#0f1219", foreground="white").grid(row=row, column=0, padx=10, pady=5, sticky='e')
        ttk.Entry(dialog, textvariable=brightness, width=10).grid(row=row, column=1, padx=10, pady=5, sticky='w')
        row += 1

        def save():
            nonlocal color_rgb
            test.update({
                'pattern_type': pattern_rev[pattern_var.get()],
                'spatial_freq_cyc_deg': sf.get(),
                'contrast': cont.get()/100.0,
                'gamma': gamma.get(),
                'velocity_deg_sec': vel.get(),
                'direction': direc.get(),
                'duration': dur.get(),
                'color_rgb': color_rgb,
                'brightness': brightness.get()
            })
            self.update_test_listbox()
            dialog.destroy()
        ttk.Button(dialog, text="OK", command=save, style='Dark.TButton').grid(row=row, column=0, columnspan=2, pady=15)
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _edit_black_screen(self, idx, item):
        dialog = tk.Toplevel(self.root)
        dialog.title("Редактировать чёрный экран")
        dialog.configure(bg="#0f1219")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(dialog, text="Длительность (с):", background="#0f1219", foreground="white").pack(padx=20, pady=10)
        dur_var = tk.DoubleVar(value=item['duration'])
        entry = ttk.Entry(dialog, textvariable=dur_var, width=10)
        entry.pack(padx=20, pady=5)
        def save():
            item['duration'] = dur_var.get()
            self.update_test_listbox()
            dialog.destroy()
        ttk.Button(dialog, text="OK", command=save, style='Dark.TButton').pack(pady=10)
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def remove_test(self):
        idx = self.test_listbox.curselection()
        if idx:
            self.tests.pop(idx[0])
            self.update_test_listbox()

    def clear_tests(self):
        self.tests.clear()
        self.update_test_listbox()

    def update_test_listbox(self):
        self.test_listbox.delete(0, tk.END)
        for i, item in enumerate(self.tests):
            if item['type'] == 'stimulus':
                dir_str = "←" if item['direction'] == 1 else "→"
                type_short = {"sinusoidal":"син","square":"кв","sawtooth":"пил","random":"случ"}.get(item['pattern_type'], item['pattern_type'][:3])
                brightness = item.get('brightness', 100)
                desc = f"{i+1:2d}. {type_short} {item['spatial_freq_cyc_deg']:.2f} cyc/° {item['velocity_deg_sec']:.0f}°/с {item['duration']:.1f}с {dir_str} ярк:{brightness}%"
            else:
                desc = f"{i+1:2d}. Чёрный экран {item['duration']:.1f}с"
            self.test_listbox.insert(tk.END, desc)

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_dir_var.set(folder)

    # ========== НОВАЯ ЛОГИКА ПАУЗЫ ==========
    def toggle_pause(self):
        if self.paused:
            # Снимаем паузу – просто сбрасываем флаг
            self.paused = False
            self.pause_btn.config(text="⏸ Пауза")
            self.status_var.set("Возобновление...")
        else:
            # Ставим паузу – останавливаем стимул и запись
            self.display_queue.put(("STOP",))
            self.camera_queue.put(("STOP_RECORDING",))
            self.paused = True
            self.pause_btn.config(text="▶ Продолжить")
            self.status_var.set("Пауза. Нажмите 'Продолжить' для повторного запуска стимула.")

    def start_series(self):
        if not self.tests:
            messagebox.showwarning("Нет тестов", "Добавьте хотя бы один элемент")
            return
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.paused = False
        self.pending_restart_item = None
        self.status_var.set("Запуск серии...")
        thread = threading.Thread(target=self.run_series_corrected)
        thread.daemon = True
        thread.start()

    def run_series_corrected(self):
        save_dir = self.save_dir_var.get()
        prefix = self.prefix_var.get()
        pre_delay = self.pre_delay_var.get()

        try:
            self.original_brightness = sbc.get_brightness()[0]
        except:
            self.original_brightness = None

        if pre_delay > 0:
            self.status_var.set(f"Предстимульный чёрный экран: {pre_delay} с")
            self.set_brightness(0)
            time.sleep(pre_delay)

        log_path = os.path.join(save_dir, "log.txt")
        os.makedirs(save_dir, exist_ok=True)

        stim_counter = 0
        i = 0
        while i < len(self.tests):
            item = self.tests[i]

            # Если это стимул
            if item['type'] == 'stimulus':
                stim_counter += 1
                # Если есть ожидающий повтор (после паузы), используем его
                if self.pending_restart_item is not None:
                    item = self.pending_restart_item
                    self.pending_restart_item = None

                brightness = item.get('brightness', 100)
                self.set_brightness(brightness)

                dir_str = "left" if item['direction'] == 1 else "right"
                freq_str = f"{item['spatial_freq_cyc_deg']:.3f}".replace('.', '_')
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"{prefix}_stim{stim_counter:03d}_{freq_str}_{dir_str}_{timestamp}.avi"
                video_path = os.path.join(save_dir, video_filename)

                self.status_var.set(f"Стимул {stim_counter}/{len([t for t in self.tests if t['type']=='stimulus'])} (ярк {brightness}%)")

                # Запускаем запись
                self.camera_queue.put(("START_RECORDING", video_path))
                time.sleep(0.3)

                params = {
                    'spatial_freq_cyc_deg': item['spatial_freq_cyc_deg'],
                    'contrast': item['contrast'],
                    'gamma': item['gamma'],
                    'velocity_deg_sec': item['velocity_deg_sec'],
                    'direction': item['direction'],
                    'pattern_type': item['pattern_type'],
                    'color_rgb': item['color_rgb']
                }
                self.display_queue.put(("START", params))

                # Ждём длительность с возможностью паузы
                start_time = time.time()
                elapsed = 0.0
                paused_here = False
                while elapsed < item['duration']:
                    time.sleep(0.1)
                    elapsed = time.time() - start_time
                    if self.paused:
                        # Пользователь нажал паузу – останавливаем стимул и запись
                        self.display_queue.put(("STOP",))
                        self.camera_queue.put(("STOP_RECORDING",))
                        self.pending_restart_item = item.copy()
                        self.status_var.set("Пауза. Нажмите 'Продолжить' для повторного запуска стимула.")
                        # Ждём снятия паузы
                        while self.paused:
                            time.sleep(0.1)
                        # Пауза снята – повторяем этот же стимул
                        paused_here = True
                        break

                if paused_here:
                    # Идём на следующую итерацию внешнего цикла с тем же i
                    continue

                # Стимул завершился без паузы
                self.display_queue.put(("STOP",))
                self.camera_queue.put(("STOP_RECORDING",))
                time.sleep(0.2)

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now().isoformat()} | Стимул {stim_counter}: тип={item['pattern_type']}, "
                            f"частота={item['spatial_freq_cyc_deg']:.3f}, скорость={item['velocity_deg_sec']:.1f}, "
                            f"направление={'влево' if item['direction']==1 else 'вправо'}, контраст={item['contrast']*100:.0f}%, "
                            f"гамма={item['gamma']:.2f}, длит={item['duration']}с, яркость={brightness}%, видео={video_filename}\n")
                i += 1

            else:
                # Чёрный экран
                self.set_brightness(0)

                video_filename = f"{prefix}_black_{i+1:03d}.avi"
                video_path = os.path.join(save_dir, video_filename)

                self.status_var.set(f"Чёрный экран {i+1}/{len(self.tests)}: {item['duration']} с")
                self.camera_queue.put(("START_RECORDING", video_path))
                time.sleep(0.3)

                # Ждём длительность (паузу на чёрном экране можно добавить по желанию)
                time.sleep(item['duration'])

                self.camera_queue.put(("STOP_RECORDING",))
                time.sleep(0.2)

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now().isoformat()} | Чёрный экран: длительность={item['duration']}с, видео={video_filename}\n")
                i += 1

            # Небольшая пауза между элементами
            time.sleep(0.1)

        if self.original_brightness is not None:
            self.set_brightness(self.original_brightness)

        self.pause_btn.config(state='disabled')
        self.start_btn.config(state='normal')
        self.status_var.set("✅ Серия завершена")

    # Заменяем старый run_series на новый
    run_series = run_series_corrected

    def update_preview_loop(self):
        w = self.preview_canvas.winfo_width()
        h = self.preview_canvas.winfo_height()
        if w > 10 and h > 10:
            spatial_freq = self.spatial_freq.get()
            contrast = self.contrast.get() / 100.0
            gamma = self.gamma.get()
            vel_deg_sec = self.velocity.get() * self.direction.get()
            pattern_type = self.pattern_type.get()
            color_rgb = self.color_rgb
            brightness = self.stimulus_brightness.get() / 100.0

            period_px = spatial_freq_to_period_px(spatial_freq, self.screen_width_deg, w)
            if period_px < 2:
                period_px = 2
            vel_px_sec = (vel_deg_sec / self.screen_width_deg) * w
            dt = 0.05
            self._preview_phase += vel_px_sec * dt
            if period_px > 0:
                self._preview_phase %= period_px
            else:
                self._preview_phase = 0.0

            img_array = generate_grating_frame(w, h, period_px, self._preview_phase, contrast, gamma, color_rgb, pattern_type)
            img_array = (img_array * brightness).astype(np.uint8)
            img = Image.fromarray(img_array)
            self.preview_img = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor='nw', image=self.preview_img)
        self.root.after(50, self.update_preview_loop)

    def process_video(self):
        try:
            while True:
                msg = self.video_queue.get_nowait()
                if msg[0] == "FRAME":
                    _, img_bytes, w, h = msg
                    img = Image.frombytes('RGB', (w, h), img_bytes)
                    canvas_w = self.cam_canvas.winfo_width()
                    canvas_h = self.cam_canvas.winfo_height()
                    if canvas_w > 10 and canvas_h > 10:
                        img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
                        img_w, img_h = img.size
                        x = (canvas_w - img_w) // 2
                        y = (canvas_h - img_h) // 2
                        self.cam_canvas.delete("all")
                        self.cam_img = ImageTk.PhotoImage(img)
                        self.cam_canvas.create_image(x, y, anchor='nw', image=self.cam_img)
                elif msg[0] == "FILENAME":
                    _, fname = msg
                    self.status_var.set(f"Запись: {fname}")
                elif msg[0] == "STOP":
                    self.status_var.set("Запись остановлена")
        except queue.Empty:
            pass
        self.root.after(50, self.process_video)

    def on_close(self):
        if self.original_brightness is not None:
            self.set_brightness(self.original_brightness)
        self.display_queue.put(("QUIT",))
        self.camera_queue.put(("QUIT",))
        time.sleep(0.5)
        self.display_proc.terminate()
        self.camera_proc.terminate()
        self.display_proc.join()
        self.camera_proc.join()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    mp.freeze_support()
    app = ModernMainWindow()
    app.run()