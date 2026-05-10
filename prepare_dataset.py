import os
import shutil
import pandas as pd

# Load CSV
df = pd.read_csv("train.csv")

src_folder = "train_images"
dst_folder = "dataset/train"

# Create class folders
for i in range(5):
    os.makedirs(f"{dst_folder}/{i}", exist_ok=True)

# Move images into class folders
for _, row in df.iterrows():
    img_name = row['id_code'] + ".png"
    label = str(row['diagnosis'])

    src = os.path.join(src_folder, img_name)
    dst = os.path.join(dst_folder, label, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("✅ Dataset prepared!")