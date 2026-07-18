import os
import shutil

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
FILES_TO_DELETE = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]

# Delete dataset folder
def delete_dataset():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
        print(f"🗑️ Deleted folder: {DATASET_DIR}")
    else:
        print("⚠️ Dataset folder not found.")

# Delete .npy files
def delete_npy_files():
    for file_name in FILES_TO_DELETE:
        file_path = os.path.join(ROOT_DIR, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Deleted file: {file_path}")
        else:
            print(f"⚠️ File not found: {file_path}")

if __name__ == "__main__":
    delete_dataset()
    delete_npy_files()
    print("✅ Cleanup complete. Only essential files remain.")
