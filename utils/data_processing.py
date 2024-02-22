
import rasterio
from rasterio.transform import from_origin
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import math

def load_geotiff(image_path):
    with rasterio.open(image_path) as src:
        # Load all 7 bands
        image = src.read([1, 2, 3, 4, 5, 6, 7])
    return image, src.transform

def geo_to_img_coords(lat, lon, transform):
    # Convert latitude and longitude to image pixel coordinates
    row, col = ~transform * (lon, lat)
    return int(row), int(col)

def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def rotate_bbox(bbox, angle, image_size):
    # Calculate the rotated bounding box
    cx, cy = image_size[0] // 2, image_size[1] // 2
    new_bbox = []
    for x, y in [(bbox[0], bbox[1]), (bbox[2], bbox[3])]:
        new_x = math.cos(angle) * (x - cx) - math.sin(angle) * (y - cy) + cx
        new_y = math.sin(angle) * (x - cx) + math.cos(angle) * (y - cy) + cy
        new_bbox.extend([new_x, new_y])
    return new_bbox

def preprocess_image(image_path, bbox_lat_lon, angle):
    # Load the GeoTiff image and its transform
    image, transform = load_geotiff(image_path)

    # Convert the geographic bbox to image coordinates
    bbox_img_coords = [geo_to_img_coords(lat, lon, transform) for lat, lon in bbox_lat_lon]

    # Convert to PIL Image for rotation
    image_pil = Image.fromarray(image.transpose(1, 2, 0))
    image_rotated = rotate_image(image_pil, angle)

    # Rotate the bounding box
    bbox_rotated = rotate_bbox(bbox_img_coords, angle, image_pil.size)

    # Normalize and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalization if necessary
    ])
    return transform(image_rotated), bbox_rotated

# Example Usage
image_tensor, transformed_bbox = preprocess_image('path_to_geotiff_image.tif', bbox_lat_lon=[[53.5455, 108.2831], [53.4434, 108.4664]], angle=30)
