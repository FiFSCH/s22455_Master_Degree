import os
import xml.etree.ElementTree as ET

from PIL import Image
from torchvision import transforms

# Validation and test dataset
standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Light augmentation
basic_augment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Heavy augmentation
heavy_augment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=30, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def apply_augmentation(image_path, label, classes, is_training=True):
    image = custom_loader(image_path)
    images = []

    images.append(standard_transform(image))

    if is_training:
        images.append(basic_augment(image))
        images.append(heavy_augment(image))

    return images, classes.index(label)

def custom_loader(path):
    image = Image.open(path).convert("RGB")
    xml_path = os.path.splitext(path)[0] + '.xml'

    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find('object')
    bndbox = obj.find('bndbox')
    xmin, ymin, xmax, ymax = [int(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]

    # Crop to bbox
    image = image.crop((xmin, ymin, xmax, ymax))

    return image
