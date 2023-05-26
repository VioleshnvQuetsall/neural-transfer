import os
import torch

from PIL import Image
from torchvision.utils import save_image

from models import *
from datasets import *
from utils import *


os.chdir(os.path.dirname(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

opt = parse_options('configs/generate.yaml')

input_shape = tuple(
    map(opt.__getattr__, ['channels', 'img_height', 'img_width']))
n_residual_blocks = opt.n_residual_blocks


def generate(model, image, resize, detransform=True):
    # preprocess image
    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if resize is not None:
        transform.append(transforms.Resize(
            resize, transforms.InterpolationMode.BICUBIC, antialias=True))
    transform = transforms.Compose(transform)

    image = transform(image).unsqueeze(0)

    # generate
    with torch.no_grad():
        fake = model(image)
    if detransform:
        fake = (fake * 0.5 + 0.5).clamp_(0, 1)
    return fake


def image_gen():
    # parse image_path
    if os.path.isfile(opt.image_path):
        image_list = [opt.image_path]
    elif os.path.isdir(opt.image_path):
        image_list = os.listdir(opt.image_path)
    else:
        raise FileNotFoundError(
            f'File or Dictionary not found: {opt.image_path}')

    # generator type to save memory
    for image_path in image_list:
        yield os.path.basename(image_path), Image.open(image_path)

def get_output_name(image_name):
    output_name = os.path.join(opt.output_dir, image_name)
    i = 0
    name, ext = os.path.splitext(image_name)
    while os.path.exists(output_name):
        i += 1
        repeat_name = f'{name}-{i}{ext}'
        output_name = os.path.join(opt.output_dir, repeat_name)
    return output_name


def generate_main():
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    weights = torch.tensor(opt.weights) / sum(opt.weights)

    # choose model
    G_AB = GeneratorResNet(input_shape, n_residual_blocks).to(device)
    G_BA = GeneratorResNet(input_shape, n_residual_blocks).to(device)

    G_AB.load_state_dict(torch.load(opt.model_path[0], map_location=device))
    G_BA.load_state_dict(torch.load(opt.model_path[1], map_location=device))

    if opt.label not in 'AB':
        raise ValueError(f'Invalid label: {opt.label}')
    model = {'A': G_AB, 'B': G_BA}[opt.label]

    # generate loop
    for image_name, image in image_gen():
        fakes = []
        for resize in opt.resizes:
            fake = generate(model, image, resize, detransform=True)
            fakes.append(fake)
        shape = fakes[-1].shape[-2:]
        t = transforms.Resize(shape, transforms.InterpolationMode.BICUBIC,
                               antialias=True)
        fakes = [t(f) * w
                 for f, w in zip(fakes, weights)]
        fake = torch.stack(fakes).sum(axis=0)

        output_name = get_output_name(image_name)
        save_image(fake, output_name)
        print(f'save {output_name}')


if __name__ == '__main__':
    generate_main()
