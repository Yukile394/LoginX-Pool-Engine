"""
╔══════════════════════════════════════════════════════════════════╗
║            LoginX-Pool-Engine  —  Bilardo Analiz Motoru          ║
║         Akademik / Eğitim Amaçlı Fizik Simülatörü               ║
║                  github.com/EnigmasHack                          ║
╚══════════════════════════════════════════════════════════════════╝

Temel Fizik:
  - Top tespiti   : OpenCV HoughCircles
  - Yansıma Yasası: θ_gidiş = θ_geliş  (Elastik Çarpışma, Sürtünmesiz)
  - Bant koordinat sistemi: 4 kenar → Sol, Sağ, Üst, Alt
"""

import cv2
import numpy as np
import sys
import os

# ─────────────────────────────────────────────────────────────────
# SABITLER  (gerekirse dışarıdan da verilebilir)
# ─────────────────────────────────────────────────────────────────
TRAJ_COLOR      = (180,  20, 180)   # Pembe (BGR)
BALL_COLOR      = (  0, 255,  80)   # Yeşil
HOLE_COLOR      = ( 20,  20, 220)   # Kırmızı
CENTER_COLOR    = (255, 255,   0)   # Sarı
TRAJ_THICKNESS  = 3
BALL_THICKNESS  = 2
MAX_BOUNCES     = 3                 # Kaç banttan sekme hesaplansın
TRAJ_LENGTH_FAC = 2.5               # Çizgi uzunluk çarpanı (masa boyutuna göre)
ARROW_TIP_LEN   = 18


# ─────────────────────────────────────────────────────────────────
# YARDIMCI: Görüntü Ön İşleme
# ─────────────────────────────────────────────────────────────────
def preprocess(img: np.ndarray) -> np.ndarray:
    """
    HoughCircles öncesi kontrast artırma + bulanıklaştırma.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (9, 9), 2)
    gray  = cv2.equalizeHist(gray)          # Kontrast normalize
    return gray


# ─────────────────────────────────────────────────────────────────
# YARDIMCI: Delik Pozisyonları Üret  (6 standart bilardo deliği)
# ─────────────────────────────────────────────────────────────────
def get_hole_positions(h: int, w: int, margin_fac: float = 0.03) -> list:
    """
    Gerçek bilardo masası kenarlarında 6 cep konumu döndürür.
    margin_fac: köşe boşluğu oranı (0–1)
    """
    mx = int(w * margin_fac)
    my = int(h * margin_fac)
    mx2 = w - mx
    my2 = h - my
    mid_x = w // 2
    mid_y = h // 2

    return [
        (mx,   my),    # Sol-Üst
        (mx2,  my),    # Sağ-Üst
        (mx,   my2),   # Sol-Alt
        (mx2,  my2),   # Sağ-Alt
        (mid_x, my),   # Orta-Üst
        (mid_x, my2),  # Orta-Alt
    ]


# ─────────────────────────────────────────────────────────────────
# YARDIMCI: Yansıma Yasası ile Bant Sekmesi
# ─────────────────────────────────────────────────────────────────
def reflect_direction(dx: float, dy: float,
                      wall: str) -> tuple:
    """
    Geliş yönü (dx, dy) ve çarpılan duvar için gidiş yönünü döndürür.
    wall: 'left' | 'right' | 'top' | 'bottom'
    """
    if wall in ('left', 'right'):
        dx = -dx            # Yatay bileşen ters
    else:
        dy = -dy            # Dikey bileşen ters
    return dx, dy


def segment_wall_intersect(px, py, dx, dy,
                            x0, y0, x1, y1,
                            wall: str) -> tuple | None:
    """
    (px,py)'den (dx,dy) yönünde ışının,
    [x0,y0]→[x1,y1] duvar segmentiyle kesişme noktasını bulur.
    Bulamazsa None döner.
    """
    # Parametrik: P + t*(d)  ve  W + s*(w1-w0)
    wx = x1 - x0
    wy = y1 - y0
    denom = dx * wy - dy * wx
    if abs(denom) < 1e-9:
        return None          # Paralel

    t = ((x0 - px) * wy - (y0 - py) * wx) / denom
    s = ((x0 - px) * dy - (y0 - py) * dx) / denom

    if t > 1e-3 and 0 <= s <= 1:
        ix = px + t * dx
        iy = py + t * dy
        return (ix, iy, t, wall)
    return None


def trace_trajectory(start_x: float, start_y: float,
                     dir_x: float, dir_y: float,
                     canvas_w: int, canvas_h: int,
                     max_bounces: int = MAX_BOUNCES) -> list:
    """
    Başlangıç noktasından verilen yönde, bantlardan sekme yasasıyla
    yörünge noktaları listesi döndürür.

    Döndürülen: [(x0,y0), (x1,y1), …]  köşe noktaları listesi
    """
    # Bant kenarları (küçük içe çekim ile masa görünümü)
    pad = 1
    walls = {
        'left'  : (pad,      0,       pad,      canvas_h),
        'right' : (canvas_w-pad, 0,   canvas_w-pad, canvas_h),
        'top'   : (0,    pad,          canvas_w, pad),
        'bottom': (0,    canvas_h-pad, canvas_w, canvas_h-pad),
    }

    path   = [(start_x, start_y)]
    cx, cy = start_x, start_y
    dx, dy = dir_x, dir_y

    # Yön vektörünü normalize et
    mag = np.hypot(dx, dy)
    if mag < 1e-9:
        return path
    dx /= mag
    dy /= mag

    step = max(canvas_w, canvas_h) * TRAJ_LENGTH_FAC   # ışın uzunluğu

    for _ in range(max_bounces + 1):
        # Bütün duvarlarla kesişimleri bul → en yakınını seç
        hits = []
        for wname, (wx0, wy0, wx1, wy1) in walls.items():
            res = segment_wall_intersect(cx, cy, dx * step, dy * step,
                                         wx0, wy0, wx1, wy1, wname)
            if res is not None:
                hits.append(res)

        if not hits:
            # Dışarı çıkıyor, en son noktayı ekle
            path.append((cx + dx * step, cy + dy * step))
            break

        # En yakın duvarı seç (t parametresi ile)
        hits.sort(key=lambda h: h[2])
        ix, iy, t_hit, wall_hit = hits[0]

        path.append((ix, iy))

        if _ == max_bounces:
            break

        # Yönü yansıt
        dx, dy = reflect_direction(dx, dy, wall_hit)
        cx, cy = ix, iy

    return path


# ─────────────────────────────────────────────────────────────────
# YARDIMCI: Yön Oku Çiz
# ─────────────────────────────────────────────────────────────────
def draw_arrow(img, pt1, pt2, color, thickness):
    """OpenCV arrowedLine sarmalayıcısı (tip_length dinamik)."""
    dist = np.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
    if dist < 1:
        return
    tip = max(0.05, min(0.3, ARROW_TIP_LEN / dist))
    cv2.arrowedLine(img,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    color, thickness,
                    tipLength=tip,
                    line_type=cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# YARDIMCI: Etiket Yaz  (arka planlı, okunaklı)
# ─────────────────────────────────────────────────────────────────
def put_label(img, text, pos, font_scale=0.48,
              fg=(255,255,255), bg=(0,0,0), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(pos[0]), int(pos[1])
    cv2.rectangle(img, (x-2, y-th-2), (x+tw+2, y+baseline+2), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# ANA MOTOR
# ─────────────────────────────────────────────────────────────────
def detect_balls(gray: np.ndarray, img_h: int, img_w: int) -> np.ndarray | None:
    """
    HoughCircles ile topları tespit eder.
    Masanın boyutuna göre min/max yarıçap otomatik hesaplanır.
    """
    min_dim = min(img_h, img_w)

    # Yarıçap aralığı: masanın %1 ile %6'sı arasında
    min_r = max(5,  int(min_dim * 0.012))
    max_r = max(30, int(min_dim * 0.07))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=min_r * 2,
        param1=70,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r
    )
    return circles


def draw_holes(canvas: np.ndarray, holes: list):
    """Delikleri çiz."""
    for i, (hx, hy) in enumerate(holes):
        cv2.circle(canvas, (hx, hy), 14, HOLE_COLOR, -1)
        cv2.circle(canvas, (hx, hy), 14, (255,255,255), 1)
        put_label(canvas, f"D{i+1}", (hx+16, hy+5),
                  fg=(200, 200, 255), bg=(40,0,40))


def draw_balls_and_trajectories(canvas: np.ndarray,
                                 circles: np.ndarray,
                                 holes: list,
                                 img_h: int, img_w: int):
    """
    Her top için:
      1. Daire + merkez nokta çiz
      2. En yakın deliğe doğru yörünge + bant sekmeleri hesapla & çiz
      3. Açı etiketini göster
    """
    balls = np.round(circles[0, :]).astype(int)

    for idx, (bx, by, br) in enumerate(balls):
        # ── Top Görselleştirme ──────────────────────────────────
        cv2.circle(canvas, (bx, by), br, BALL_COLOR, BALL_THICKNESS, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), 3,  CENTER_COLOR, -1, cv2.LINE_AA)
        put_label(canvas, f"T{idx+1}", (bx + br + 4, by - 4),
                  fg=(80, 255, 80), bg=(0, 30, 0))

        # ── En Yakın Deliği Bul ─────────────────────────────────
        if not holes:
            continue
        dists = [np.hypot(bx - hx, by - hy) for hx, hy in holes]
        nearest_idx = int(np.argmin(dists))
        hx, hy = holes[nearest_idx]

        # ── Doğrudan Vuruş Yönü ────────────────────────────────
        dx = hx - bx
        dy = hy - by
        mag = np.hypot(dx, dy)
        if mag < 1e-9:
            continue

        # Açı hesapla (x-ekseni referans, derece)
        angle_deg = np.degrees(np.arctan2(-dy, dx))   # ekran koord. düzelt

        # ── Bant Sekme Yörüngesi ───────────────────────────────
        path = trace_trajectory(float(bx), float(by),
                                 dx, dy,
                                 img_w, img_h,
                                 max_bounces=MAX_BOUNCES)

        # Yörüngeyi çiz (köşelerle)
        for seg_i in range(len(path) - 1):
            p1 = path[seg_i]
            p2 = path[seg_i + 1]
            # İlk segment → oklı
            if seg_i == 0:
                draw_arrow(canvas, p1, p2, TRAJ_COLOR, TRAJ_THICKNESS)
            else:
                cv2.line(canvas,
                         (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])),
                         TRAJ_COLOR,
                         max(1, TRAJ_THICKNESS - 1),
                         cv2.LINE_AA)
            # Sekme noktası işareti
            if seg_i > 0:
                cv2.circle(canvas, (int(p1[0]), int(p1[1])), 5,
                           (255, 160, 0), -1, cv2.LINE_AA)
                put_label(canvas, f"B{seg_i}",
                          (int(p1[0])+6, int(p1[1])-6),
                          fg=(255,200,100), bg=(40,20,0), font_scale=0.4)

        # ── Açı Etiketi ────────────────────────────────────────
        label_pos = (bx + int(np.cos(np.radians(angle_deg)) * (br + 20)),
                     by - int(np.sin(np.radians(angle_deg)) * (br + 20)))
        put_label(canvas, f"{angle_deg:.1f}°", label_pos,
                  fg=(255, 100, 255), bg=(30, 0, 30))


# ─────────────────────────────────────────────────────────────────
# MASA SINIRLARINI GÖRSELLEŞTIR
# ─────────────────────────────────────────────────────────────────
def draw_table_border(canvas: np.ndarray):
    h, w = canvas.shape[:2]
    pad = 4
    cv2.rectangle(canvas, (pad, pad), (w-pad, h-pad),
                  (20, 130, 20), 3, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# LOGO / BAŞLIK DAMGASI
# ─────────────────────────────────────────────────────────────────
def draw_watermark(canvas: np.ndarray):
    h, w = canvas.shape[:2]
    text = "LoginX-Pool-Engine  |  github: EnigmasHack"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.45
    thick  = 1
    (tw, th), _ = cv2.getTextSize(text, font, fscale, thick)
    x = w - tw - 8
    y = h - 8
    cv2.rectangle(canvas, (x-3, y-th-4), (x+tw+3, y+4), (0,0,0), -1)
    cv2.putText(canvas, text, (x, y), font, fscale,
                (120, 220, 120), thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────────────
def analyze(input_path: str, output_path: str) -> bool:
    """
    Bilardo görüntüsünü analiz eder; sonucu output_path'e kaydeder.
    Başarılıysa True, aksi hâlde False döner.
    """
    print(f"[LoginX-Pool-Engine] Görüntü okunuyor: {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        print(f"[HATA] Görüntü okunamadı: {input_path}")
        return False

    h, w = img.shape[:2]
    print(f"[INFO] Görüntü boyutu: {w}x{h}")

    # Ön işleme
    gray = preprocess(img)

    # Canvas (orijinal görüntü üzerine çizim)
    canvas = img.copy()

    # Masa sınırı
    draw_table_border(canvas)

    # Delikler
    holes = get_hole_positions(h, w)
    draw_holes(canvas, holes)

    # Top Tespiti
    circles = detect_balls(gray, h, w)

    if circles is None:
        print("[UYARI] Hiç top tespit edilemedi. Parametreler gevşetiliyor…")
        # İkinci deneme: daha gevşek
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=15, param1=50, param2=20,
            minRadius=4, maxRadius=int(min(h,w) * 0.1)
        )

    if circles is not None:
        n = circles.shape[1]
        print(f"[INFO] {n} top tespit edildi.")
        draw_balls_and_trajectories(canvas, circles, holes, h, w)
    else:
        print("[UYARI] Top tespit edilemedi. Ham görüntü kaydediliyor.")
        put_label(canvas, "Top Tespit Edilemedi",
                  (w//4, h//2), font_scale=1.0,
                  fg=(0,0,255), bg=(255,255,255), thickness=2)

    # Filigran
    draw_watermark(canvas)

    # Kaydet
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ok = cv2.imwrite(output_path, canvas)
    if ok:
        print(f"[OK] Sonuç kaydedildi: {output_path}")
    else:
        print(f"[HATA] Görüntü kaydedilemedi: {output_path}")
    return ok


# ─────────────────────────────────────────────────────────────────
# CLI GİRİŞ NOKTASI
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    inp  = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    outp = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"
    success = analyze(inp, outp)
    sys.exit(0 if success else 1)
      
