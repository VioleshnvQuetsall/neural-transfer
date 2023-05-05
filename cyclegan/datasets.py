from pathlib import Path
from PIL import Image
from random import randint

import os

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        if transform is None:
            self.transform = None
        else:
            self.transform = transforms.Compose(transform)

        images_path = Path(path)
        images_list = list(map(str, images_path.glob('*.jpg')))
        self.images = images_list

    def __getitem__(self, index):
        image_path = self.images[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image  # CHW

    def sample(self, item):
        if type(item) == int:
            index = item
        else:
            for i, a in enumerate(self.images):
                if a.endswith(item):
                    index = i
                    break
            else:
                raise ValueError(f'{item} is not in images')
        image = self[index]
        return image.reshape(1, *image.shape)

    def __len__(self):
        return len(self.images)


class ImagePairDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode='train'):
        if transform is None:
            self.transform = None
        else:
            self.transform = transforms.Compose(transform)

        self.unaligned = unaligned

        path_A = Path(os.path.join(root, f'{mode}A'))
        path_B = Path(os.path.join(root, f'{mode}B'))
        self.images_A_list = list(sorted(map(str, path_A.glob('*.jpg'))))
        self.images_B_list = list(sorted(map(str, path_B.glob('*.jpg'))))

    def __getitem__(self, index):
        image_A = Image.open(self.images_A_list[index % len(self.images_A_list)])
        image_B = Image.open(self.images_B_list[randint(0, len(self.images_B_list) - 1)
                                                if self.unaligned
                                                else index % len(self.images_B_list)])

        if self.transform is not None:
            item_A = self.transform(image_A)
            item_B = self.transform(image_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.images_A_list), len(self.images_B_list))


def default_transform(height, width):
    return [
        transforms.Resize(int(height * 1.12), transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]