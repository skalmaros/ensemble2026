import os
import glob
import time
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from scipy.signal import find_peaks, medfilt
from scipy.interpolate import interp1d


PAPER_WIDTH_MM = 279.4
PAPER_HEIGHT_MM = 215.9
PAPER_SPEED_MM_S = 25.0
MM_PER_MV = 10.0
TARGET_FS = 500
SEGMENT_DURATION_S = 2.5

TRAIN_WIDTH = 1024
TRAIN_HEIGHT = 768
MASK_WIDTH = 2200
MASK_HEIGHT = 1700

PIXELS_PER_MM_X = MASK_WIDTH / PAPER_WIDTH_MM
PIXELS_PER_MM_Y = MASK_HEIGHT / PAPER_HEIGHT_MM

MODEL_PATH = "unet_best.pth"
TEST_DIR = "data/train_ready_files"
OUTPUT_FILE = "submission.npz"
BATCH_SIZE = 32

PAPER_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

LEAD_NAMES_ORDERED = [lead for row in PAPER_LAYOUT for lead in row]
NUM_ROWS = len(PAPER_LAYOUT)
NUM_COLS = len(PAPER_LAYOUT[0])

LEAD_NAME_SUBMISSION = {
    "I": "I", "II": "II", "III": "III",
    "aVR": "AVR", "aVL": "AVL", "aVF": "AVF",
    "V1": "V1", "V2": "V2", "V3": "V3",
    "V4": "V4", "V5": "V5", "V6": "V6",
}

EXPECTED_LEADS = list(LEAD_NAME_SUBMISSION.values())
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


def generate_masks_batch(image_paths, model, device):
    tensors = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Could not load: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TRAIN_WIDTH, TRAIN_HEIGHT))
        tensors.append(img.transpose(2, 0, 1).astype(np.float32) / 255.0)

    batch = torch.from_numpy(np.stack(tensors)).to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()

    masks = []
    for pred in preds:
        mask = (pred > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (MASK_WIDTH, MASK_HEIGHT),
                          interpolation=cv2.INTER_NEAREST)
        masks.append(mask)
    return masks



def find_left_margin(mask):
    h, w = mask.shape
    left_quarter = mask[:, :w // 4]
    binary = left_quarter > 127
    col_counts = binary.sum(axis=0)
    has_signal = col_counts > 0
    if has_signal.sum() < 2:
        return 0
    return int(np.where(has_signal)[0][0])


def split_mask(mask, num_rows=3, num_cols=4):
    h, w = mask.shape

    profil_y = np.sum(mask > 127, axis=1)
    if profil_y.max() == 0:

        empty = np.zeros((h // num_rows, COL_WIDTH_PX), dtype=np.uint8)
        baselines = [h // (2 * num_rows) + i * (h // num_rows) for i in range(num_rows)]
        cuts = [i * (h // num_rows) for i in range(num_rows + 1)]
        return [empty] * (num_rows * num_cols), baselines, cuts, COL_WIDTH_PX

    file, _ = find_peaks(profil_y, distance=h // 8,
                         prominence=np.max(profil_y) * 0.05)
    file_sorted = sorted(file)

    if len(file_sorted) >= num_rows + 1:
        base_line = list(file_sorted[:num_rows])
        lower = (file_sorted[num_rows - 1] + file_sorted[num_rows]) // 2
    elif len(file_sorted) >= num_rows:
        base_line = list(file_sorted[:num_rows])
        lower = h
    elif len(file_sorted) == 2:
        base_line = list(file_sorted[:2])
        gap = file_sorted[1] - file_sorted[0]
        base_line.append(min(file_sorted[1] + gap, h - 10))
        lower = h
    elif len(file_sorted) == 1:
        step = h // (num_rows + 1)
        base_line = [step * (i + 1) for i in range(num_rows)]
        lower = h
    else:
        step = h // (num_rows + 1)
        base_line = [step * (i + 1) for i in range(num_rows)]
        lower = h

    cutting_line_y = [0]
    for i in range(len(base_line) - 1):
        cutting_line_y.append((base_line[i] + base_line[i + 1]) // 2)
    cutting_line_y.append(lower)

    margin_left = find_left_margin(mask)

    segments = []
    for row in range(len(cutting_line_y) - 1):
        for col in range(num_cols):
            x0 = margin_left + col * COL_WIDTH_PX
            x1 = min(x0 + COL_WIDTH_PX, w)
            segments.append(mask[cutting_line_y[row]:cutting_line_y[row + 1],
                                 x0:x1])

    return segments, base_line, cutting_line_y, COL_WIDTH_PX



def digitize_crop(crop, baseline_row):
    h, w = crop.shape
    if h == 0 or w == 0:
        return np.zeros(int(SEGMENT_DURATION_S * TARGET_FS), dtype=np.float32)

    binary = crop > 127
    col_counts = binary.sum(axis=0)
    has_signal = col_counts > 0
    num_samples = int(round(SEGMENT_DURATION_S * TARGET_FS))

    if has_signal.sum() < 2:
        return np.zeros(num_samples, dtype=np.float32)

    signal_cols = np.where(has_signal)[0]
    x_start = signal_cols[0]
    x_end = signal_cols[-1]
    signal_width = x_end - x_start + 1

    if signal_width < 2:
        return np.zeros(num_samples, dtype=np.float32)

    y_mid = np.full(signal_width, np.nan)

    for x in range(x_start, x_end + 1):
        if has_signal[x]:
            white = np.where(binary[:, x])[0]
            if len(white) > 0:
                y_mid[x - x_start] = np.mean(white)

    valid = ~np.isnan(y_mid)
    x_valid = np.where(valid)[0]

    if len(x_valid) < 2:
        return np.zeros(num_samples, dtype=np.float32)

    f_mid = interp1d(x_valid, y_mid[valid], kind="linear", fill_value="extrapolate")
    mid_cont = f_mid(np.arange(signal_width))

    voltage = (baseline_row - mid_cont) / (MM_PER_MV * PIXELS_PER_MM_Y)

    old_x = np.linspace(0, 1, len(voltage))
    new_x = np.linspace(0, 1, num_samples)
    return interp1d(old_x, voltage, kind="linear")(new_x).astype(np.float32)


def validate_submission(npz_path, expected_records):
    print(f"\n{'='*60}")
    print("VALIDATE  SUBMISSION")
    print(f"{'='*60}")

    data = np.load(npz_path)
    keys = list(data.keys())
    total_expected = len(expected_records) * len(EXPECTED_LEADS)

    print(f"KEY: {len(keys)} (EXPECTED {total_expected})")

    errors = []
    for key in keys:
        arr = data[key]
        if arr.dtype != np.float16:
            errors.append(f"{key}: dtype={arr.dtype}")
        if arr.ndim != 1 or len(arr) == 0:
            errors.append(f"{key}: shape={arr.shape}")

    print(f"\nExample:")
    for key in sorted(keys)[:5]:
        arr = data[key]
        print(f"  {key}: {arr.shape} {arr.dtype} [{arr.min():.3f}, {arr.max():.3f}]")

    lengths = [len(data[k]) for k in keys]
    print(f"\nLenght: {min(lengths)}-{max(lengths)} samples "
          f"({min(lengths)/TARGET_FS:.2f}-{max(lengths)/TARGET_FS:.2f}s)")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:15]:
            print(f"  {e}")
    else:
        print(f"\nSucess validation")

    print(f"{'='*60}")
    data.close()



if __name__ == "__main__":
    t_start = time.time()
    print("=" * 50)
    print("ECG SUBMISSION PIPELINE")
    print("=" * 50)

    model = load_model(MODEL_PATH)
    device = next(model.parameters()).device
    print(f"DEVICE: {device}")

    patterns = [os.path.join(TEST_DIR, f"*.{e}") for e in ("png", "jpg", "jpeg")]
    image_paths = sorted(p for pat in patterns for p in glob.glob(pat))
    n = len(image_paths)
    record_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    print(f"Imagines: {n}")

    if not image_paths:
        raise FileNotFoundError(f"No images in {TEST_DIR}")

    submission = {}
    errors = []

    for batch_start in range(0, n, BATCH_SIZE):
        batch_paths = image_paths[batch_start:batch_start + BATCH_SIZE]
        batch_names = [os.path.splitext(os.path.basename(p))[0] for p in batch_paths]

        try:
            masks = generate_masks_batch(batch_paths, model, device)
        except Exception as e:
            for name in batch_names:
                errors.append((name, str(e)))
            continue

        for mask, rec_name in zip(masks, batch_names):
            try:
                segments, base_line, cutting_line, col_width_px = split_mask(
                    mask, num_rows=NUM_ROWS, num_cols=NUM_COLS
                )

                for i, segment in enumerate(segments):
                    if i >= len(LEAD_NAMES_ORDERED):
                        break

                    row = i // NUM_COLS
                    if row < len(base_line):
                        bl = base_line[row] - cutting_line[row]
                    else:
                        bl = segment.shape[0] // 2

                    sig = digitize_crop(segment, bl)

                    lead_sub = LEAD_NAME_SUBMISSION[LEAD_NAMES_ORDERED[i]]
                    key = f"{rec_name}_{lead_sub}"
                    submission[key] = sig.astype(np.float16)

            except Exception as e:
                errors.append((rec_name, str(e)))

        del masks
        done = min(batch_start + BATCH_SIZE, n)
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        print(f"  [{done}/{n}] {rate:.1f} img/s | ETA: {eta:.0f}s")

    print(f"\nSaving {len(submission)} keys...")
    np.savez_compressed(OUTPUT_FILE, **submission)

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    total = time.time() - t_start
    print(f"\nKKeys:  {len(submission)}")
    print(f"Files:    {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print(f"Time:    {total:.0f}s ({total/60:.1f} min)")
    if errors:
        print(f"Error:   {len(errors)}")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    validate_submission(OUTPUT_FILE, record_names)