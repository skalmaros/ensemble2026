import cv2
import os
import glob

folder_zdjec = "data/train_wyprostowane"
folder_masek = "data/train_masks"

print("Rozpoczynam skanowanie plików pod kątem uszkodzeń...")

# Szukamy wszystkich zdjęć w folderze
search_pattern = os.path.join(folder_zdjec, "*.png")
image_files = glob.glob(search_pattern)

usunięte_pliki = 0

for sciezka_zdjecia in image_files:
    # Próbujemy wczytać plik
    img = cv2.imread(sciezka_zdjecia)
    
    # Jeśli OpenCV zwróci None, plik jest uszkodzony!
    if img is None:
        nazwa_pliku = os.path.basename(sciezka_zdjecia)
        print(f" -> Znaleziono uszkodzony plik: {nazwa_pliku}. Usuwam...")
        
        # 1. Usuwamy uszkodzone zdjęcie
        os.remove(sciezka_zdjecia)
        
        # 2. Usuwamy odpowiadającą mu maskę (jeśli istnieje)
        sciezka_maski = os.path.join(folder_masek, nazwa_pliku)
        if os.path.exists(sciezka_maski):
            os.remove(sciezka_maski)
            
        usunięte_pliki += 1

print(f"\nGotowe! Skanowanie zakończone. Usunięto {usunięte_pliki} uszkodzonych par.")