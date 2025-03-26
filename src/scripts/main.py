import constants
from utils import remove_all_backgrounds, download_all_logos, ImageGrouper

import pandas as pd
from keras.src.applications.vgg16 import VGG16
from keras.src.applications.resnet_v2 import ResNet50V2

# read the data from the parquet file
df = pd.read_parquet("../logos.snappy.parquet", engine="pyarrow")

# download every logo and store it in the LOGOS_FOLDER from constants
download_all_logos(df, out_dest=constants.LOGOS_FOLDER)

remove_all_backgrounds(constants.LOGOS_FOLDER)

# from experiments, VGG16 does all right for this task
model_VGG16 = VGG16(weights='imagenet', include_top=False, pooling='avg')
# model_ResNet50 = ResNet50V2(weights='imagenet', include_top=False, pooling='avg')

# use the class defined in utils to cleanly handle logo grouping
image_grouper = ImageGrouper(model_VGG16)

image_grouper.group_images_with_cosine_similarity(
    constants.LOGOS_FOLDER,
    out_folder=constants.OUTPUT_FOLDER,
    similarity_threshold=0.825
)
'''
image_grouper.group_images_with_clustering(
    constants.LOGOS_FOLDER,
    out_folder=constants.OUTPUT_FOLDER,
    n_clusters=100
)
'''