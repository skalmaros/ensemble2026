import cv2
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing


def process_single_image(image_path, output_folder):
    file_name = os.path.basename(image_path)
    save_path = os.path.join(output_folder, file_name)

    try:
        img = cv2.imread(image_path)
        if img is None:
            return f"BŁĄD: Nie można wczytać pliku {file_name}"

        h, w = img.shape[:2]

        scale = 0.25
        small_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
        angles = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -45 < angle < 45:
                    angles.append(angle)

        if not angles:
            cv2.imwrite(save_path, img)
            return f"Zapisano oryginał (brak siatki): {file_name}"

        median_angle = np.median(angles)

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, scale=1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        cv2.imwrite(save_path, rotated_img)
        return f"Wyprostowano: {file_name} (Kąt: {median_angle:.2f}°)"

    except Exception as e:
        return f"BŁĄD KRYTYCZNY w {file_name}: {str(e)}"


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Utworzono folder wyjściowy: {output_folder}")

    search_pattern = os.path.join(input_folder, "*.png")
    image_files = glob.glob(search_pattern)

    total_files = len(image_files)
    if total_files == 0:
        print("Nie znaleziono żadnych plików .png!")
        return

    max_workers = min(8, multiprocessing.cpu_count())
    print(f"Znaleziono {total_files} plików. Używam {max_workers} procesów. Start...\n")

    processed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, img_path, output_folder): img_path for img_path in image_files}

        for future in as_completed(futures):
            processed_count += 1

            try:
                result_msg = future.result(timeout=60)
            except TimeoutError:
                result_msg = "BŁĄD: Timeout (>60s)"
            except Exception as e:
                result_msg = f"BŁĄD: {str(e)}"

            if processed_count % 20 == 0 or processed_count == total_files:
                print(f"[{processed_count}/{total_files}] {result_msg}")

    print(f"\nGotowe! Przetworzono {processed_count}/{total_files} plików.")


if __name__ == '__main__':
    folder_wejsciowy = "data/test"
    folder_wyjsciowy = "data/test_wyprostowane"

    process_folder(folder_wejsciowy, folder_wyjsciowy)