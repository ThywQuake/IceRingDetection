# Ice Ring Detection

## Main Task

1. Collect images of icerings
2. Annotating the places of icerings in images
3. From step 1 and 2, we can generate two kinds of files:
   1. Images containing icerings
   2. Annotation files showing where their are
4. Build an architecture to train a model to detect icerings in images based on Faster R-CNN
5. Train, test and evaluate the model
6. Apply the model to detect icerings in broader range of images
7. Exact the information of icerings from the images
8. Analyze the information of icerings

## 1. Collect images of icerings

- Image Sources: 'MODIS/061/MOD09A1' on Google Earth Engine
- Sample Information: in "archive.csv" file
- Executive Code Generator: in "data_process.ipynb" file, function "gee_code_generator"

## 2. Annotating the places of icerings in images

1. Use gee_code_generator to generate the code to collect images from Google Earth Engine
2. At the meantime, use tools in Google Earth Engine to annotate the places of icerings in images
3. Download the images
4. Copy and convert the annotation we drew in GEE to the format of Faster R-CNN(xml format), using the function "text_to_xml" in "data_process.ipynb" file

