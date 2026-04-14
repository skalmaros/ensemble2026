import cv2
import os
import glob

image_folder = "data/train_ready_files"
mask_folder = "data/train_masks"


search_pattern = os.path.join(image_folder, "*.png")
image_files = glob.glob(search_pattern)

deleted_files = 0

for image_path in image_files:
    img = cv2.imread(image_path)
    
    if img is None:
        nazwa_pliku = os.path.basename(image_path)
        print(f" -> Broken file: {nazwa_pliku}.")
        
        os.remove(image_path)
        
        sciezka_maski = os.path.join(mask_folder, nazwa_pliku)
        if os.path.exists(sciezka_maski):
            os.remove(sciezka_maski)
            
        deleted_files += 1

print(f"\nDone. Deleted {deleted_files} broken files.")