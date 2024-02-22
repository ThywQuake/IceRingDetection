import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

from utils.data_processing import *
import random

class IceRingDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, split='train', split_ratio=0.8):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = os.listdir(image_dir)
        _,self.transform = load_geotiff(image_dir + self.images[0])

        # Splitting the dataset
        random.shuffle(self.images)
        split_index = int(len(self.images) * split_ratio)
        if split == 'train':
            self.images = self.images[:split_index]
        else:  # split == 'test'
            self.images = self.images[split_index:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load and preprocess the image
        img_path = os.path.join(self.image_dir, self.images[idx])
        annotation_path = os.path.join(self.annotation_dir, self.images[idx].replace('.tif', '.xml'))
        
        bbox_lat_lon, angle = self.parse_annotation(annotation_path)
        image, bbox = preprocess_image(img_path, bbox_lat_lon, angle)

        # Convert bbox to the format expected by Faster R-CNN
        target = {'boxes': torch.as_tensor([bbox], dtype=torch.float32),
                  'labels': torch.as_tensor([1], dtype=torch.int64)}  # Assuming only one class

        return image, target

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Extract coordinates
        coordinates = root.find('.//coordinates').text.strip().split(' ')
        lat_lons = [list(map(float, coord.split(','))) for coord in coordinates]

        # Convert the lat/lon coordinates to image coordinates
        img_coords = [geo_to_img_coords(lat, lon, self.transform) for lat, lon in lat_lons]

        # Convert the polygon coordinates to a bounding box [xmin, ymin, xmax, ymax]
        x_coords = [coord[0] for coord in img_coords]
        y_coords = [coord[1] for coord in img_coords]
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

        # Placeholder for rotation angle (if applicable)
        angle = 0  # Modify as needed

        return bbox, angle