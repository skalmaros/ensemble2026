import json
import cv2
import numpy as np
import os
import glob

def generate_mask_from_json(json_path, output_mask_path):
    # 1. Wczytanie danych z pliku JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Pobranie wymiarów oryginalnego zdjęcia
    # Jeśli w JSON brakuje tych danych, ustawiamy domyślne na podstawie Twojego pliku
    width = data.get('width', 2200)
    height = data.get('height', 1700)

    # 3. Utworzenie pustego, całkowicie czarnego obrazu (tła)
    # Rozmiar w numpy podajemy jako (wysokość, szerokość)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 4. Rysowanie idealnego sygnału EKG na podstawie współrzędnych
    leads = data.get('leads', [])
    for lead in leads:
        pixels = lead.get('plotted_pixels', [])
        
        if not pixels:
            continue

        # Przekształcamy współrzędne zmiennoprzecinkowe z JSON na liczby całkowite (piksele)
        # Z formatu JSON [y, x] robimy krotki (x, y) wymagane przez OpenCV
        points = [(int(round(p[1])), int(round(p[0]))) for p in pixels]

        # Rysujemy ciągłą linię łączącą kolejne punkty sygnału
        for i in range(1, len(points)):
            pt1 = points[i-1]
            pt2 = points[i]
            
            # Parametry: obraz, punkt_start, punkt_koniec, kolor (255 = biały), grubość linii
            # Grubość 2 lub 3 pikseli jest optymalna dla treningu sieci U-Net
            cv2.line(mask, pt1, pt2, 255, thickness=2)

    # 5. Zapisanie gotowej maski na dysku
    cv2.imwrite(output_mask_path, mask)


def process_all_jsons(json_folder, output_masks_folder):
    # Tworzymy folder docelowy, jeśli nie istnieje
    if not os.path.exists(output_masks_folder):
        os.makedirs(output_masks_folder)
        print(f"Utworzono folder na maski: {output_masks_folder}")

    # Wyszukujemy wszystkie pliki .json w folderze
    search_pattern = os.path.join(json_folder, "*.json")
    json_files = glob.glob(search_pattern)

    if not json_files:
        print("Nie znaleziono plików .json w podanym folderze!")
        return

    print(f"Znaleziono {len(json_files)} plików .json. Generowanie masek...\n")

    for idx, json_path in enumerate(json_files, 1):
        # Pobieramy nazwę pliku (np. "ecg_train_0001.json")
        base_name = os.path.basename(json_path)
        # Zmieniamy rozszerzenie na .png dla maski (np. "ecg_train_0001.png")
        mask_filename = base_name.replace(".json", ".png")
        
        output_path = os.path.join(output_masks_folder, mask_filename)
        
        print(f"[{idx}/{len(json_files)}] Tworzenie maski: {mask_filename}")
        generate_mask_from_json(json_path, output_path)

    print("\nGotowe! Wszystkie maski zostały wygenerowane.")



folder_z_plikami_json = "data/train"
folder_na_gotowe_maski = "data/train_maski"

process_all_jsons(folder_z_plikami_json, folder_na_gotowe_maski)