import constants
from utils import download_logo, group_images_with_clustering

import pandas as pd
from keras.src.applications.vgg16 import VGG16
from keras.src.applications.resnet_v2 import ResNet50V2

# read the data from the parquet file
df = pd.read_parquet("../logos.snappy.parquet", engine="pyarrow")

# download every logo and store it in the LOGOS_FOLDER from constants
for row in df.itertuples():
    domain = row[1]

    logo_file = download_logo(domain, dest=constants.LOGOS_FOLDER)
    if logo_file is None:
        print(f"Error downloading logo for domain {domain}")
        continue

# from experiments, VGG16 does all right for this task
model_VGG16 = VGG16(weights='imagenet', include_top=False, pooling='avg')
#model_ResNet50 = ResNet50V2(weights='imagenet', include_top=False, pooling='avg')


# call this function from utils to group images into clusters
group_images_with_clustering(constants.LOGOS_FOLDER, model=model_VGG16, out_folder=constants.OUTPUT_FOLDER, n_clusters=10)

