import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk


def to_tk(img: np.ndarray, max_side=380) -> ImageTk.PhotoImage:
    if img.ndim == 2:
        pil = Image.fromarray(img)
    else:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil.thumbnail((max_side, max_side))
    return ImageTk.PhotoImage(pil)


def get_hist(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.bincount(img.ravel(), minlength=256)


def hist_to_image(hist: np.ndarray, w=256, h=120) -> np.ndarray:
    hist = hist.astype(np.float64)
    # log1p смягчает пики на гистограмме, чтобы она влезала в панель
    vis = np.log1p(hist) if hist.max() > 0 else hist
    if vis.max() > 0:
        vis = vis / vis.max()
    img = np.full((h, w), 255, np.uint8)
    for i in range(w):
        if vis[i] > 0:
            img[int(h - vis[i] * h):, i] = 0
    return img


def gray_metrics(gray: np.ndarray) -> dict:
    g = gray.astype(np.float64)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p_nz = p[p > 0]
    return {
        "min": int(gray.min()),
        "max": int(gray.max()),
        "mean": float(g.mean()),
        "std": float(g.std()),
        "entropy": float(-np.sum(p_nz * np.log2(p_nz))),
    }


def fmt_metrics(name: str, m: dict) -> str:
    return (f"{name:<30} min={m['min']:3d} max={m['max']:3d} "
            f"mean={m['mean']:6.2f} std={m['std']:6.2f} H={m['entropy']:.3f}\n")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    mse = np.mean((a_f - b_f) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def linear_contrast(gray: np.ndarray) -> np.ndarray:
    f = gray.astype(np.float32)
    fmin, fmax = f.min(), f.max()
    if fmax - fmin < 1e-6:
        return gray.copy()
    return np.clip(255.0 * (f - fmin) / (fmax - fmin), 0, 255).astype(np.uint8)


def equalize_hist(gray: np.ndarray) -> np.ndarray:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist / hist.sum())
    lut = np.clip(np.round(255.0 * cdf), 0, 255).astype(np.uint8)
    return lut[gray]


def equalize_color_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.merge([equalize_hist(c) for c in cv2.split(bgr)])


def equalize_color_hsv(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return cv2.cvtColor(cv2.merge([h, s, equalize_hist(v)]), cv2.COLOR_HSV2BGR)


def _sorted_windows(gray: np.ndarray, ksize: int) -> np.ndarray:
    # sliding_window_view создаёт представление без копирования,
    # что критично для скорости при больших окнах
    pad = ksize // 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    H, W = gray.shape
    win = np.lib.stride_tricks.sliding_window_view(padded, (ksize, ksize))
    win = win.reshape(H, W, ksize * ksize)
    return np.sort(win, axis=-1)


def order_statistic_filter(gray, ksize=3, rank=None):
    N = ksize * ksize
    if rank is None:
        rank = (N - 1) // 2
    assert 0 <= rank < N
    return _sorted_windows(gray, ksize)[:, :, rank].astype(np.uint8)


def median_filter(gray, k):
    return order_statistic_filter(gray, k, (k * k - 1) // 2)


def min_filter(gray, k):
    return order_statistic_filter(gray, k, 0)


def max_filter(gray, k):
    return order_statistic_filter(gray, k, k * k - 1)


def add_salt_pepper(img: np.ndarray, prob=0.05, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    n = out.size
    n_noise = int(n * prob)
    coords = rng.choice(n, n_noise, replace=False)
    vals = rng.choice([0, 255], n_noise).astype(np.uint8)
    flat = out.reshape(-1)
    flat[coords] = vals
    return flat.reshape(out.shape)


class TabContrast(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._photos = []

        ctrl = ttk.Frame(self, padding=6)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(ctrl, text="Линейное контрастирование",
                   command=self.do_linear).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Эквализация гистограммы",
                   command=self.do_equalize).pack(side=tk.LEFT, padx=4)
        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(ctrl, text="Цвет: RGB (поканально)",
                   command=self.do_color_rgb).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Цвет: HSV (только V)",
                   command=self.do_color_hsv).pack(side=tk.LEFT, padx=4)
        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(ctrl, text="Сравнить всё",
                   command=self.compare_all).pack(side=tk.LEFT, padx=4)

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.lbl_src = self._img_panel(top, "Исходное")
        self.lbl_dst = self._img_panel(top, "Результат")

        bot = ttk.Frame(self)
        bot.pack(side=tk.TOP, fill=tk.X, padx=6)
        self.lbl_hsrc = self._hist_panel(bot, "Гистограмма: исходное")
        self.lbl_hdst = self._hist_panel(bot, "Гистограмма: результат")

        self.txt = tk.Text(self, height=8, font=("Courier New", 10))
        self.txt.pack(fill=tk.X, padx=6, pady=(0, 6))

    def _img_panel(self, parent, title):
        fr = ttk.LabelFrame(parent, text=title)
        fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        lbl = ttk.Label(fr, anchor=tk.CENTER)
        lbl.pack(fill=tk.BOTH, expand=True)
        return lbl

    def _hist_panel(self, parent, title):
        fr = ttk.LabelFrame(parent, text=title)
        fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        lbl = ttk.Label(fr, anchor=tk.CENTER)
        lbl.pack(fill=tk.BOTH, expand=True)
        return lbl

    def _show(self, label, img):
        ph = to_tk(img)
        self._photos.append(ph)
        label.configure(image=ph)

    def _show_hist(self, label, hist):
        ph = to_tk(hist_to_image(hist), max_side=512)
        self._photos.append(ph)
        label.configure(image=ph)

    def _log(self, s):
        self.txt.insert(tk.END, s)
        self.txt.see(tk.END)

    def _show_pair(self, gray_src, gray_dst):
        self._show(self.lbl_dst, gray_dst)
        self._show_hist(self.lbl_hdst, get_hist(gray_dst))

    def refresh(self):
        g = self.app.gray
        if g is None:
            return
        self._show(self.lbl_src, self.app.original_bgr)
        self._show_hist(self.lbl_hsrc, get_hist(g))

    def do_linear(self):
        if self.app.gray is None:
            return
        res = linear_contrast(self.app.gray)
        self._show_pair(self.app.gray, res)
        self._log("\n=== ЛИНЕЙНОЕ КОНТРАСТИРОВАНИЕ ===\n")
        self._log(fmt_metrics("До", gray_metrics(self.app.gray)))
        self._log(fmt_metrics("После", gray_metrics(res)))

    def do_equalize(self):
        if self.app.gray is None:
            return
        res = equalize_hist(self.app.gray)
        self._show_pair(self.app.gray, res)
        self._log("\n=== ЭКВАЛИЗАЦИЯ ГИСТОГРАММЫ ===\n")
        self._log(fmt_metrics("До", gray_metrics(self.app.gray)))
        self._log(fmt_metrics("После", gray_metrics(res)))

    def do_color_rgb(self):
        if self.app.original_bgr is None:
            return
        res = equalize_color_rgb(self.app.original_bgr)
        self._show(self.lbl_dst, res)
        self._show_hist(self.lbl_hdst, get_hist(res))
        self._log("\n[ЦВЕТ, RGB] поканальная эквализация R, G, B\n")

    def do_color_hsv(self):
        if self.app.original_bgr is None:
            return
        res = equalize_color_hsv(self.app.original_bgr)
        self._show(self.lbl_dst, res)
        self._show_hist(self.lbl_hdst, get_hist(res))
        self._log("\n[ЦВЕТ, HSV] эквализация только яркости V\n")

    def compare_all(self):
        if self.app.gray is None:
            return
        lin = linear_contrast(self.app.gray)
        eq = equalize_hist(self.app.gray)
        self._show(self.lbl_dst, eq)
        self._show_hist(self.lbl_hdst, get_hist(eq))

        self._log("\n" + "=" * 72 + "\n")
        self._log("СРАВНЕНИЕ ПОВЫШЕНИЯ КОНТРАСТА (полутоновое)\n")
        self._log("=" * 72 + "\n")
        self._log(fmt_metrics("Исходное", gray_metrics(self.app.gray)))
        self._log(fmt_metrics("Линейное контрастирование", gray_metrics(lin)))
        self._log(fmt_metrics("Эквализация", gray_metrics(eq)))

        if self.app.original_bgr is not None:
            g_rgb = cv2.cvtColor(equalize_color_rgb(self.app.original_bgr),
                                 cv2.COLOR_BGR2GRAY)
            g_hsv = cv2.cvtColor(equalize_color_hsv(self.app.original_bgr),
                                 cv2.COLOR_BGR2GRAY)
            self._log("\n" + "-" * 72 + "\n")
            self._log("СРАВНЕНИЕ ЦВЕТНОЙ ЭКВАЛИЗАЦИИ\n")
            self._log("-" * 72 + "\n")
            self._log(fmt_metrics("RGB (поканально)", gray_metrics(g_rgb)))
            self._log(fmt_metrics("HSV (только V)", gray_metrics(g_hsv)))
        self._log("\n")


class TabOrderStats(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._photos = []
        self.noisy_gray = None

        ctrl = ttk.Frame(self, padding=6)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(ctrl, text="Добавить шум соль/перец",
                   command=self.add_noise).pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="  Окно:").pack(side=tk.LEFT)
        self.ksize_var = tk.IntVar(value=3)
        ttk.Combobox(ctrl, textvariable=self.ksize_var,
                     values=[3, 5, 7, 9], width=3,
                     state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(ctrl, text="Медианный",
                   command=self.run_median).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl, text="Минимума",
                   command=self.run_min).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl, text="Максимума",
                   command=self.run_max).pack(side=tk.LEFT, padx=3)

        ttk.Label(ctrl, text="  rank:").pack(side=tk.LEFT)
        self.rank_var = tk.IntVar(value=0)
        ttk.Spinbox(ctrl, from_=0, to=80, textvariable=self.rank_var,
                    width=4).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl, text="k-я статистика",
                   command=self.run_kth).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl, text="Сравнить все",
                   command=self.compare_all).pack(side=tk.LEFT, padx=10)

        grid = ttk.Frame(self)
        grid.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.panels = {}
        titles = [
            ("src", "Исходное (без шума)"),
            ("noisy", "Зашумлённое (соль/перец)"),
            ("median", "Медианный фильтр"),
            ("min", "Фильтр минимума"),
            ("max", "Фильтр максимума"),
            ("kth", "k-я порядковая статистика"),
        ]
        for i, (key, title) in enumerate(titles):
            row, col = divmod(i, 3)
            fr = ttk.LabelFrame(grid, text=title)
            fr.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            grid.rowconfigure(row, weight=1)
            grid.columnconfigure(col, weight=1)
            lbl = ttk.Label(fr, anchor=tk.CENTER)
            lbl.pack(fill=tk.BOTH, expand=True)
            self.panels[key] = lbl

        self.txt = tk.Text(self, height=8, font=("Courier New", 10))
        self.txt.pack(fill=tk.X, padx=6, pady=(0, 6))

    def _show(self, key, img):
        ph = to_tk(img)
        self._photos.append(ph)
        self.panels[key].configure(image=ph)

    def _log(self, s):
        self.txt.insert(tk.END, s)
        self.txt.see(tk.END)

    def refresh(self):
        self.noisy_gray = None
        if self.app.gray is not None:
            self._show("src", self.app.gray)
        for k in ("noisy", "median", "min", "max", "kth"):
            self.panels[k].configure(image="")

    def add_noise(self):
        if self.app.gray is None:
            messagebox.showinfo("Нет изображения", "Сначала загрузите изображение")
            return
        self.noisy_gray = add_salt_pepper(self.app.gray, prob=0.05, seed=42)
        self._show("noisy", self.noisy_gray)
        self._log("\nДобавлен импульсный шум «соль и перец» (5%)\n")

    def _need_noisy(self):
        if self.noisy_gray is None:
            messagebox.showinfo("Нет шума",
                                "Сначала нажмите «Добавить шум соль/перец»")
            return False
        return True

    def run_median(self):
        if not self._need_noisy():
            return
        k = self.ksize_var.get()
        res = median_filter(self.noisy_gray, k)
        self._show("median", res)
        self._log(f"Медианный (окно {k}): PSNR = {psnr(self.app.gray, res):.2f} dB\n")

    def run_min(self):
        if not self._need_noisy():
            return
        k = self.ksize_var.get()
        res = min_filter(self.noisy_gray, k)
        self._show("min", res)
        self._log(f"Минимума (окно {k}): PSNR = {psnr(self.app.gray, res):.2f} dB\n")

    def run_max(self):
        if not self._need_noisy():
            return
        k = self.ksize_var.get()
        res = max_filter(self.noisy_gray, k)
        self._show("max", res)
        self._log(f"Максимума (окно {k}): PSNR = {psnr(self.app.gray, res):.2f} dB\n")

    def run_kth(self):
        if not self._need_noisy():
            return
        k = self.ksize_var.get()
        N = k * k
        rank = int(self.rank_var.get())
        if not 0 <= rank < N:
            messagebox.showerror("Неверный ранг",
                                 f"Для окна {k}×{k} rank ∈ [0, {N-1}]")
            return
        res = order_statistic_filter(self.noisy_gray, k, rank)
        self._show("kth", res)
        self._log(f"k-я статистика (окно {k}, rank={rank}): "
                  f"PSNR = {psnr(self.app.gray, res):.2f} dB\n")

    def compare_all(self):
        if not self._need_noisy():
            return
        k = self.ksize_var.get()
        med = median_filter(self.noisy_gray, k)
        mn = min_filter(self.noisy_gray, k)
        mx = max_filter(self.noisy_gray, k)
        self._show("median", med)
        self._show("min", mn)
        self._show("max", mx)

        self._log("\n" + "=" * 72 + "\n")
        self._log(f"СРАВНЕНИЕ ФИЛЬТРОВ, окно {k}×{k}\n")
        self._log("=" * 72 + "\n")
        self._log(f"{'Метод':<28}{'PSNR vs эталон':>18}{'std':>10}\n")
        self._log("-" * 72 + "\n")
        for name, img in [("Зашумлённое", self.noisy_gray),
                          ("Медианный", med),
                          ("Минимума", mn),
                          ("Максимума", mx)]:
            self._log(f"{name:<28}{psnr(self.app.gray, img):>16.2f} dB"
                      f"{img.std():>10.2f}\n")
        self._log("\n")


class MainApp:
    def __init__(self, root):
        self.root = root
        root.title("Лаб. №2 — вариант 5 — БГУ, компьютерная графика")
        root.geometry("1300x920")

        self.original_bgr = None
        self.gray = None

        top = ttk.Frame(root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Загрузить изображение",
                   command=self.load_image).pack(side=tk.LEFT, padx=4)
        self.lbl_info = ttk.Label(top, text="изображение не загружено",
                                  foreground="gray")
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab_contrast = TabContrast(nb, self)
        self.tab_stats = TabOrderStats(nb, self)
        nb.add(self.tab_contrast, text="Гистограммы и контраст")
        nb.add(self.tab_stats, text="Порядковые статистики")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("Ошибка", "Не удалось открыть файл")
            return
        h, w = bgr.shape[:2]
        max_side = 600
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self.original_bgr = bgr
        self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.lbl_info.configure(
            text=f"{os.path.basename(path)}   {bgr.shape[1]}×{bgr.shape[0]}")
        self.tab_contrast.refresh()
        self.tab_stats.refresh()


if __name__ == "__main__":
    root = tk.Tk()
    MainApp(root)
    root.mainloop()
