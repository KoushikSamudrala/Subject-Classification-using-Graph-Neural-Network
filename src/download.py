import os
import urllib.request
import tarfile

def download_and_extract(url, download_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    tgz_path = os.path.join(extract_path, 'cora.tgz')
    if not os.path.exists(tgz_path):
        print("Downloading...")
        urllib.request.urlretrieve(url, tgz_path)
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Done.")

if __name__ == "__main__":
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    download_and_extract(url, "data/raw", "data/raw")