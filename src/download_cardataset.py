import urllib.request as urllib
import tarfile
import glob
import os

def download_file(url, path):
    file_name = url.split('/')[-1]
    u = urllib.urlopen(url)
    f = open(path + file_name, 'wb')
    meta = u.info()
    file_size = int(meta.get_all("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))
    
    file_size_dl = 0
    block_sz = 8192
    
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status)
    
    f.close()


def extract(tar_url, extract_path='.'):
    print(tar_url)
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])

if __name__ == "__main__":
    training_images = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
    test_images = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
    test_kit = 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'

    # download all three files and unzip them
    download_file(test_kit, '../raw_data/')
    download_file(training_images, '../raw_data/')
    download_file(test_images, '../raw_data/')

    if not os.path.exists('dataset/'):
        os.makedirs('dataset/')
    
    for fname in glob.glob('../raw_data/*.tgz'):
        extract(fname, extract_path='dataset/')
        