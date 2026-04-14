import json
import cv2
import numpy as np
import os
import glob

def generate_mask_from_json(json_path, output_mask_path):

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    width = data.get('width', 2200)
    height = data.get('height', 1700)


    mask = np.zeros((height, width), dtype=np.uint8)


    leads = data.get('leads', [])
    for lead in leads:
        pixels = lead.get('plotted_pixels', [])
        
        if not pixels:
            continue

        points = [(int(round(p[1])), int(round(p[0]))) for p in pixels]


        for i in range(1, len(points)):
            pt1 = points[i-1]
            pt2 = points[i]
            

            cv2.line(mask, pt1, pt2, 255, thickness=2)


    cv2.imwrite(output_mask_path, mask)


def process_all_jsons(json_folder, output_masks_folder):

    if not os.path.exists(output_masks_folder):
        os.makedirs(output_masks_folder)
        print(f"A folder has been created: {output_masks_folder}")


    search_pattern = os.path.join(json_folder, "*.json")
    json_files = glob.glob(search_pattern)

    if not json_files:
        print("No .json files found in the specified folder")
        return

    print(f"Found {len(json_files)} .json files. Generating masks...\n")

    for idx, json_path in enumerate(json_files, 1):
        base_name = os.path.basename(json_path)
        mask_filename = base_name.replace(".json", ".png")
        
        output_path = os.path.join(output_masks_folder, mask_filename)
        
        print(f"[{idx}/{len(json_files)}] Creating mask: {mask_filename}")
        generate_mask_from_json(json_path, output_path)

    print("\nDone.")



folder_with_json_masks = "data/train"
folder_with_ready_masks = "data/train_masks"

process_all_jsons(folder_with_json_masks, folder_with_ready_masks)