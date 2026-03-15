import os
import cv2
import wfdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from scipy.signal import find_peaks, correlate, medfilt
from scipy.interpolate import interp1d

IMAGE_PATH = "data/train_wyprostowane/ecg_train_0041.png"
REC_BASE = "data/train/ecg_train_0041"
SCIEZKA_MODEL = "unet_best.pth"

MASK_WIDTH = 2200
MASK_HEIGHT = 1700
PIXELS_PER_MM_X = MASK_WIDTH / 279.4
PIXELS_PER_MM_Y = MASK_HEIGHT / 215.9
PAPER_SPEED_MM_S = 25.0
MM_PER_MV = 10.0
TARGET_FS = 500
SEGMENT_DURATION_S = 2.5
EXPECTED_SAMPLES = 1250
TRAIN_WIDTH = 1024
TRAIN_HEIGHT = 768

PAPER_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
LEAD_NAMES_ORDERED = [lead for row in PAPER_LAYOUT for lead in row]
NUM_ROWS = len(PAPER_LAYOUT)
NUM_COLS = len(PAPER_LAYOUT[0])
COL_WIDTH_PX = int(SEGMENT_DURATION_S * PAPER_SPEED_MM_S * PIXELS_PER_MM_X)


def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    return model.to(device).eval()


def generate_mask(image_path, model):
    device = next(model.parameters()).device
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (TRAIN_WIDTH, TRAIN_HEIGHT))
    tensor = (torch.from_numpy(img_resized.transpose(2, 0, 1))
              .float().unsqueeze(0).to(device) / 255.0)
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    mask = (pred > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (MASK_WIDTH, MASK_HEIGHT), interpolation=cv2.INTER_NEAREST)


def clean_mask(mask):
    """
    Delikatny postprocessing — usuwa tylko małe izolowane plamy.
    Nie rusza samego sygnału.
    """
    binary = (mask > 127).astype(np.uint8)

    # Tylko connected components — usuń małe izolowane plamy
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = 50  # bardzo małe plamy (tekst, kropki)

    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1

    return cleaned * 255


def find_left_margin(maska):
    h, w = maska.shape
    left_quarter = maska[:, :w // 4]
    binary = left_quarter > 127
    col_counts = binary.sum(axis=0)
    has_signal = col_counts > 0
    if has_signal.sum() < 2:
        return 0
    return np.where(has_signal)[0][0]


def smart_split_by_baselines(maska, num_rows=3, num_cols=4):
    h, w = maska.shape

    profil_y = np.sum(maska > 127, axis=1)
    piki, _ = find_peaks(profil_y, distance=h // 8, prominence=np.max(profil_y) * 0.05)
    piki_sorted = sorted(piki)

    if len(piki_sorted) >= num_rows + 1:
        linie_bazowe = list(piki_sorted[:num_rows])
        dolna_granica = (piki_sorted[num_rows - 1] + piki_sorted[num_rows]) // 2
    elif len(piki_sorted) >= num_rows:
        linie_bazowe = list(piki_sorted[:num_rows])
        dolna_granica = h
    else:
        linie_bazowe = list(piki_sorted)
        dolna_granica = h

    linie_ciecia = [0]
    for i in range(len(linie_bazowe) - 1):
        linie_ciecia.append((linie_bazowe[i] + linie_bazowe[i + 1]) // 2)
    linie_ciecia.append(dolna_granica)

    margin_left = find_left_margin(maska)

    wycinki = []
    for row in range(len(linie_ciecia) - 1):
        for col in range(num_cols):
            x_start = margin_left + col * COL_WIDTH_PX
            x_end = min(x_start + COL_WIDTH_PX, w)
            wycinki.append(maska[linie_ciecia[row]:linie_ciecia[row + 1],
                                 x_start:x_end])

    return wycinki, linie_bazowe, linie_ciecia, margin_left, COL_WIDTH_PX


def digitize_crop(crop, baseline_row):
    """
    Outer edge: bierz krawędź DALSZĄ od baseline.
    - mid < baseline → sygnał powyżej → bierz top (prawdziwa amplituda dodatnia)
    - mid >= baseline → sygnał poniżej → bierz bot (prawdziwa amplituda ujemna)
    """
    h, w = crop.shape
    binary = crop > 127
    col_counts = binary.sum(axis=0)
    has_signal = col_counts > 0

    if has_signal.sum() < 2:
        width_mm = w / PIXELS_PER_MM_X
        duration_s = width_mm / PAPER_SPEED_MM_S
        num_samples = max(int(round(duration_s * TARGET_FS)), 1)
        return np.zeros(num_samples, dtype=np.float32)

    signal_cols = np.where(has_signal)[0]
    x_start = signal_cols[0]
    x_end = signal_cols[-1]
    signal_width = x_end - x_start + 1

    width_mm = signal_width / PIXELS_PER_MM_X
    duration_s = width_mm / PAPER_SPEED_MM_S
    num_samples = max(int(round(duration_s * TARGET_FS)), 1)

    y_top = np.full(signal_width, np.nan)
    y_bot = np.full(signal_width, np.nan)
    y_mid = np.full(signal_width, np.nan)

    for x in range(x_start, x_end + 1):
        if has_signal[x]:
            white = np.where(binary[:, x])[0]
            y_top[x - x_start] = white[0]
            y_bot[x - x_start] = white[-1]
            y_mid[x - x_start] = np.mean(white)

    valid = ~np.isnan(y_top)
    x_valid = np.where(valid)[0]

    if len(x_valid) < 2:
        return np.zeros(num_samples, dtype=np.float32)

    f_top = interp1d(x_valid, y_top[valid], kind="linear", fill_value="extrapolate")
    f_bot = interp1d(x_valid, y_bot[valid], kind="linear", fill_value="extrapolate")
    f_mid = interp1d(x_valid, y_mid[valid], kind="linear", fill_value="extrapolate")

    top_cont = f_top(np.arange(signal_width))
    bot_cont = f_bot(np.arange(signal_width))
    mid_cont = f_mid(np.arange(signal_width))

    y_signal = np.where(mid_cont < baseline_row, top_cont, bot_cont)

    voltage = (baseline_row - y_signal) / (MM_PER_MV * PIXELS_PER_MM_Y)

    if len(voltage) > 5:
        voltage = medfilt(voltage, kernel_size=3)

    old_x = np.linspace(0, 1, len(voltage))
    new_x = np.linspace(0, 1, num_samples)
    return interp1d(old_x, voltage, kind="linear")(new_x).astype(np.float32)


def measure_shift(gt, model_sig, max_shift_samples=300):
    min_len = min(len(gt), len(model_sig))
    if min_len < 50:
        return 0, 0.0, 0.0

    g = gt[:min_len] - np.mean(gt[:min_len])
    m = model_sig[:min_len] - np.mean(model_sig[:min_len])

    if np.std(g) < 1e-6 or np.std(m) < 1e-6:
        return 0, 0.0, 0.0

    corr = correlate(g, m, mode='full')
    mid_idx = len(m) - 1

    lo = max(mid_idx - max_shift_samples, 0)
    hi = min(mid_idx + max_shift_samples + 1, len(corr))

    best_idx = lo + np.argmax(corr[lo:hi])
    shift_samples = best_idx - mid_idx

    shift_ms = shift_samples / TARGET_FS * 1000

    if shift_samples > 0:
        aligned_m = model_sig[shift_samples:]
        aligned_g = gt[:len(aligned_m)]
    elif shift_samples < 0:
        aligned_g = gt[-shift_samples:]
        aligned_m = model_sig[:len(aligned_g)]
    else:
        aligned_g = gt[:min_len]
        aligned_m = model_sig[:min_len]

    al = min(len(aligned_g), len(aligned_m))
    if al < 50:
        return shift_samples, shift_ms, 0.0

    ag = aligned_g[:al] - np.mean(aligned_g[:al])
    am = aligned_m[:al] - np.mean(aligned_m[:al])
    if np.std(ag) < 1e-6 or np.std(am) < 1e-6:
        return shift_samples, shift_ms, 0.0

    best_corr = np.corrcoef(ag, am)[0, 1]
    return shift_samples, shift_ms, best_corr


if __name__ == "__main__":
    record_name = os.path.basename(REC_BASE)
    out_dir = f"debug_{record_name}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rekord: {record_name}")

    model = load_model(SCIEZKA_MODEL)
    mask = generate_mask(IMAGE_PATH, model)
    cv2.imwrite(f"{out_dir}/01_maska_raw.png", mask)

    mask = clean_mask(mask)
    cv2.imwrite(f"{out_dir}/01_maska_clean.png", mask)

    record = wfdb.rdrecord(REC_BASE)
    rec_leads_upper = [n.upper() for n in record.sig_name]
    print(f"Fs: {record.fs} Hz | Leads: {record.sig_name} | Próbki: {record.sig_len}")

    wycinki, linie_bazowe, linie_ciecia, margin_left, col_width_px = \
        smart_split_by_baselines(mask, num_rows=NUM_ROWS, num_cols=NUM_COLS)

    print(f"Linie bazowe: {linie_bazowe}")
    print(f"Linie cięcia: {linie_ciecia}")
    print(f"Lewy margines: {margin_left}px")
    print(f"Szerokość kolumny: {col_width_px}px (oczekiwane dla 2.5s)")

    # DIAGNOSTYKA
    print(f"\n{'='*75}")
    print("DIAGNOSTYKA POZYCJI SYGNAŁU W WYCINKACH:")
    print(f"{'='*75}")
    for i, wycinek in enumerate(wycinki):
        if i >= len(LEAD_NAMES_ORDERED):
            break
        lead_name = LEAD_NAMES_ORDERED[i]
        col = i % NUM_COLS

        binary = wycinek > 127
        col_counts = binary.sum(axis=0)
        has_signal = col_counts > 0

        if has_signal.sum() < 2:
            print(f"  {lead_name:<6} — brak sygnału")
            continue

        signal_cols = np.where(has_signal)[0]
        x_start = signal_cols[0]
        x_end = signal_cols[-1]
        signal_width = x_end - x_start + 1

        margin_l = x_start
        margin_r = wycinek.shape[1] - x_end - 1
        margin_l_ms = margin_l / PIXELS_PER_MM_X / PAPER_SPEED_MM_S * 1000

        print(f"  {lead_name:<6} kol={col} | "
              f"wycinek={wycinek.shape[1]}px | "
              f"sygnał: {x_start}-{x_end} ({signal_width}px) | "
              f"margines L={margin_l}px ({margin_l_ms:.0f}ms) R={margin_r}px")

    # CIĘCIE — wizualizacja
    fig_cut, ax_cut = plt.subplots(figsize=(14, 9))
    ax_cut.imshow(mask, cmap="gray")
    for b in linie_bazowe:
        ax_cut.axhline(y=b, color="red", linewidth=2, alpha=0.7)
    for c in linie_ciecia:
        ax_cut.axhline(y=c, color="lime", linewidth=2, linestyle="--")
    for col in range(NUM_COLS + 1):
        x = margin_left + col * col_width_px
        ax_cut.axvline(x=x, color="cyan", linewidth=2, linestyle=":")
    for i, lead in enumerate(LEAD_NAMES_ORDERED):
        row = i // NUM_COLS
        col = i % NUM_COLS
        if row < len(linie_ciecia) - 1:
            y_mid = (linie_ciecia[row] + linie_ciecia[row + 1]) // 2
            x_mid = margin_left + col * col_width_px + col_width_px // 2
            ax_cut.text(x_mid, y_mid, lead, color="yellow", fontsize=12,
                        ha="center", va="center", fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.7, pad=2))
    ax_cut.set_title(f"{record_name} | margin_left={margin_left}px, col_w={col_width_px}px")
    fig_cut.savefig(f"{out_dir}/02_ciecie.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cut)

    # KAFELKI
    fig_tiles, axes_tiles = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(20, 10))
    for i, wycinek in enumerate(wycinki):
        if i >= NUM_ROWS * NUM_COLS:
            break
        ax = axes_tiles[i // NUM_COLS][i % NUM_COLS]
        ax.imshow(wycinek, cmap="gray")
        ax.set_title(LEAD_NAMES_ORDERED[i], fontsize=12)
        ax.axis("off")
    fig_tiles.savefig(f"{out_dir}/03_kafelki.png", dpi=150, bbox_inches="tight")
    plt.close(fig_tiles)

    # PORÓWNANIE — z diagnostyką amplitudy
    fig, axes = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(22, 12))
    metrics = {}

    for i, wycinek in enumerate(wycinki):
        if i >= len(LEAD_NAMES_ORDERED):
            break
        lead_name = LEAD_NAMES_ORDERED[i]
        row = i // NUM_COLS
        col = i % NUM_COLS
        ax = axes[row][col]

        if row < len(linie_bazowe):
            bl = linie_bazowe[row] - linie_ciecia[row]
        else:
            bl = wycinek.shape[0] // 2

        sig = digitize_crop(wycinek, bl)

        start_idx = int(col * SEGMENT_DURATION_S * record.fs)
        end_idx = start_idx + int(SEGMENT_DURATION_S * record.fs)

        try:
            lead_idx = rec_leads_upper.index(lead_name.upper())
        except ValueError:
            ax.set_title(f"{lead_name} — BRAK", color="red", fontsize=9)
            continue
        if end_idx > record.sig_len:
            ax.set_title(f"{lead_name} — za krótki", color="red", fontsize=9)
            continue

        gt = record.p_signal[start_idx:end_idx, lead_idx].copy()
        ex = sig.copy()

        shift_samples, shift_ms, corr_aligned = measure_shift(gt, ex)

        min_len = min(len(gt), len(ex))
        gt_c = gt[:min_len] - np.median(gt[:min_len])
        ex_c = ex[:min_len] - np.median(ex[:min_len])

        mae = np.mean(np.abs(gt_c - ex_c))
        corr_raw = 0.0
        if np.std(gt_c) > 1e-6 and np.std(ex_c) > 1e-6:
            corr_raw = np.corrcoef(gt_c, ex_c)[0, 1]

        gt_range = np.max(gt_c) - np.min(gt_c)
        ex_range = np.max(ex_c) - np.min(ex_c)
        amp_ratio = ex_range / gt_range if gt_range > 1e-6 else 0.0

        metrics[lead_name] = {
            "MAE": mae, "r_raw": corr_raw, "r_aligned": corr_aligned,
            "shift": shift_samples, "shift_ms": shift_ms,
            "gt_range": gt_range, "ex_range": ex_range, "amp_ratio": amp_ratio,
        }

        t = np.linspace(0, SEGMENT_DURATION_S, min_len)
        ax.plot(t, gt_c, color="blue", alpha=0.6, linewidth=1, label="GT")
        ax.plot(t, ex_c, color="red", alpha=0.8, linewidth=1, linestyle="--", label="Model")

        if abs(shift_ms) > 5:
            shift_dir = "→" if shift_ms > 0 else "←"
            shift_color = "orange" if abs(shift_ms) < 100 else "red"
            ax.annotate(f"{shift_dir} {abs(shift_ms):.0f}ms",
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=9, color=shift_color, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, pad=2))

        s = "✅" if corr_raw > 0.5 else "⚠️" if corr_raw > 0.2 else "❌"
        ax.set_title(f"{lead_name} | r={corr_raw:.3f} amp={amp_ratio:.2f}x {s}", fontsize=8)
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=5)

    fig.suptitle(f"{record_name} — outer edge + clean_mask", fontsize=14)
    fig.savefig(f"{out_dir}/04_porownanie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # TABELA Z AMPLITUDĄ
    print(f"\n{'='*85}")
    print(f"{'Lead':<8} {'MAE':>8} {'r_raw':>8} {'r_align':>8} {'Shift':>10} "
          f"{'GT_rng':>8} {'EX_rng':>8} {'AmpRat':>8}")
    print(f"{'-'*85}")
    for lead in LEAD_NAMES_ORDERED:
        if lead in metrics:
            m = metrics[lead]
            s = "✅" if m["r_raw"] > 0.5 else "⚠️" if m["r_raw"] > 0.2 else "❌"
            print(f"{lead:<8} {m['MAE']:>8.4f} {m['r_raw']:>8.4f} {m['r_aligned']:>8.4f} "
                  f"{m['shift']:>+5d} ({m['shift_ms']:>+6.0f}ms) "
                  f"{m['gt_range']:>8.3f} {m['ex_range']:>8.3f} {m['amp_ratio']:>7.2f}x {s}")
    if metrics:
        avg_mae = np.mean([m["MAE"] for m in metrics.values()])
        avg_r = np.mean([m["r_raw"] for m in metrics.values()])
        avg_r_al = np.mean([m["r_aligned"] for m in metrics.values()])
        avg_shift = np.mean([m["shift_ms"] for m in metrics.values()])
        avg_amp = np.mean([m["amp_ratio"] for m in metrics.values()])
        print(f"{'-'*85}")
        print(f"{'ŚREDNIA':<8} {avg_mae:>8.4f} {avg_r:>8.4f} {avg_r_al:>8.4f} "
              f"      ({avg_shift:>+6.0f}ms) "
              f"{'':>8} {'':>8} {avg_amp:>7.2f}x")

        print(f"\n{'='*75}")
        print("ANALIZA PRZESUNIĘCIA PER KOLUMNA:")
        print(f"{'='*75}")
        for col_idx in range(NUM_COLS):
            col_leads = [LEAD_NAMES_ORDERED[row * NUM_COLS + col_idx] for row in range(NUM_ROWS)]
            col_shifts = [metrics[l]["shift_ms"] for l in col_leads if l in metrics]
            if col_shifts:
                leads_str = ", ".join(col_leads)
                avg_s = np.mean(col_shifts)
                std_s = np.std(col_shifts)
                print(f"  Kol {col_idx+1} ({leads_str}): "
                      f"avg={avg_s:+.0f}ms  std={std_s:.0f}ms  "
                      f"wartości: {[f'{s:+.0f}' for s in col_shifts]}")

    print(f"{'='*75}")
    print(f"\nPliki w {out_dir}/:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")