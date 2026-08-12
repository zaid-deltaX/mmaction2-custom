import os

directory = "./dataset/v2/val/normal"

files = sorted(
    f for f in os.listdir(directory)
    if f.lower().endswith(".mp4")
    # if f.lower().endswith(".avi")
)

for i, filename in enumerate(files, start=1):
    old_path = os.path.join(directory, filename)
    new_name = f"normal_{i:03d}.mp4"
    new_path = os.path.join(directory, new_name)

    os.rename(old_path, new_path)
    print(f"{filename} -> {new_name}")

print("Renaming completed!")