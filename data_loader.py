import requests
from pathlib import Path
import tarfile
import shutil
import random
import os


DATA_DIRECTORY = Path('data')
SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
FILE_NAME = 'cifar-10-python.tar.gz'
EXTRACTED_FOLDER = 'extracted_raw'
META_NAME = 'batches.meta'

def get_directory() -> Path:
    directory = DATA_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def get_tar_file() -> Path:
    return (get_directory() / FILE_NAME)

def get_extracted_path() -> Path:
    return get_directory() / EXTRACTED_FOLDER

def get_split_paths() -> tuple[Path, Path, Path]:
    train_path = get_directory() / 'train'
    test_path = get_directory() / 'test'
    validate_path = get_directory() / 'validate'

    if train_path.exists():
        shutil.rmtree(train_path)
    if test_path.exists():
        shutil.rmtree(test_path)
    if validate_path.exists():
        shutil.rmtree(validate_path)

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    validate_path.mkdir(parents=True, exist_ok=True)

    return (train_path, test_path, validate_path)

def get_meta_path() -> Path:
    return get_directory() / META_NAME


def download_data():
    print('Starting download')
    r = requests.get(SOURCE_URL)
    print('Saving to', FILE_NAME)
    with get_tar_file().open('wb') as file:
        file.write(r.content)
    
def extract_data():
    print('extracting data')
    with tarfile.open(get_tar_file(), 'r:gz') as file:
        file.extractall(path = get_extracted_path())


def create_split(train: int, test: int, validate: int, data: list[Path]) -> tuple[list[Path]]:
    # shuffle data
    data = list(data)
    random.shuffle(data)

    num_validate = round(len(data) * (validate / 100))
    num_test = round(len(data) * (test / 100))
    num_train = len(data) - num_validate - num_test

    validate_paths = [data.pop() for _ in range(num_validate)]
    test_paths = [data.pop() for _ in range(num_test)]
    train_paths = [data.pop() for _ in range(num_train)]

    return train_paths, test_paths, validate_paths
    
def clean_split_data():
    train_path, test_path, validate_path = get_split_paths()

    if train_path.exists():
        shutil.rmtree(train_path)
    if test_path.exists():
        shutil.rmtree(test_path)
    if validate_path.exists():
        shutil.rmtree(validate_path)

def split_data(train: int, test: int, validate: int):

    clean_split_data()

    train_path, test_path, validate_path = get_split_paths()

    meta:Path = None
    batches = []

    raw_data = get_extracted_path()
    for path in raw_data.rglob("*batch*"):
        if path.is_dir():
            continue
        
        if path.name == META_NAME:
            meta = path
        else:
            batches.append(path)

    if get_meta_path().exists():
        os.remove(get_meta_path())

    shutil.move(meta, get_directory())
    train_paths, test_paths, validate_paths = create_split(train, test, validate, batches)

    for path in train_paths:
        shutil.move(path, train_path)
    for path in test_paths:
        shutil.move(path, test_path)
    for path in validate_paths:
        shutil.move(path, validate_path)



def clean_up():
    path = get_extracted_path()
    shutil.rmtree(path)

def main():
    #download_data()
    extract_data()
    split_data(60,20,20)
    clean_up()

if __name__ == '__main__':

    # create consitent data sampleing
    random.seed(1234)

    main()





# import torch
# import torchvision
# import torchvision.transforms as transforms

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batch_size = 4

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

