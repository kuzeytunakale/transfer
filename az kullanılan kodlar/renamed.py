import os
import random

image_folders = ["C:\\Users\\Kuzey Tuna\\Desktop\\sonuncu\\Yeni klasör\\main_project_transfer\\main_project_transfer\\dataset\\test\\3-green-glass"]

for image_folder in image_folders:

    # Dizin içindeki tüm dosyaları listele
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
     # Dosyaları sırasıyla yeniden adlandır
    for idx, image_file in enumerate(sorted(image_files), start=1):
        old_path = os.path.join(image_folder, image_file)
        new_name = f"{random.randint(1, 9999999999999999999999999999999999999999999)}.jpg"  # Yeni isim
        new_path = os.path.join(image_folder, new_name)
        
        # Dosyayı yeniden adlandır
        os.rename(old_path, new_path)
        print(f"{old_path} -> {new_path}")  # Hangi dosyanın adının değiştiğini yazdır

    # Dizin içindeki tüm dosyaları listele
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    # Dosyaları sırasıyla yeniden adlandır
    for idx, image_file in enumerate(sorted(image_files), start=1):
        old_path = os.path.join(image_folder, image_file)
        new_name = f"{idx}.jpg"  # Yeni isim
        new_path = os.path.join(image_folder, new_name)
        
        # Dosyayı yeniden adlandır
        os.rename(old_path, new_path)
        print(f"{old_path} -> {new_path}")  # Hangi dosyanın adının değiştiğini yazdır