"""
Download MovieLens 100K Dataset
"""
import urllib.request
import zipfile
from pathlib import Path

def download_movielens():
    """Download and extract MovieLens 100K dataset"""
    print("Downloading MovieLens 100K dataset...")
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Download URL
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = data_dir / 'ml-100k.zip'
    
    # Download
    if not zip_path.exists():
        print("  Downloading...")
        urllib.request.urlretrieve(url, zip_path)
        print("  ✓ Download complete!")
    else:
        print("  Dataset already downloaded.")
    
    # Extract
    extract_path = data_dir / 'ml-100k'
    if not extract_path.exists():
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("  ✓ Extraction complete!")
    else:
        print("  Dataset already extracted.")
    
    # Verify files
    required_files = ['u.data', 'u.item', 'u.user']
    print("\n  Verifying files:")
    for file in required_files:
        file_path = extract_path / file
        if file_path.exists():
            print(f"    ✓ {file} found")
        else:
            print(f"    ✗ {file} MISSING!")
            return False
    
    print("\n✅ Dataset ready!")
    return True

if __name__ == "__main__":
    download_movielens()
