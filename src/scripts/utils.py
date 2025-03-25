import constants
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keras.src.applications.vgg16 import preprocess_input
from keras.src.utils import img_to_array
import requests
from PIL import Image


# obtain the logo of a domain, using Clearbit's Logo API
# the logo is saved at [dest] with the name [domain].PNG
# optionally, you can pass out_file as a parameter to specify the output file's name
def download_logo(domain, dest='.', output_file_name=None):
    response = requests.get(url=f'{constants.CLEARBIT_LOGO_API_URL}/{domain}')
    if response.status_code == 200:
        os.makedirs(dest, exist_ok=True)
        out_file_name = f'{domain}.png' if output_file_name is None else f'{output_file_name}.png'
        out_file = os.path.join(dest, out_file_name)

        with open(out_file, "wb") as file:
            file.write(response.content)
            return f'{dest}/{out_file_name}'
    else:
        return None

# delete file at filepath
def delete_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

# obtain useful information from an image using a pretrained model
def extract_features(img_path, model):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_data = img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# KMeans clustering for grouping images
def group_images_with_clustering(image_folder, model, out_folder=".", n_clusters=5, random_state=42):
    # get all PNG files in the folder
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

    # extract features for all images
    features_list = [extract_features(img, model) for img in image_files]
    features_array = np.array(features_list)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(features_array)

    # group images by cluster
    grouped_images = {}
    for i, cluster in enumerate(clusters):
        if cluster not in grouped_images:
            grouped_images[cluster] = []
        grouped_images[cluster].append(image_files[i])

    # save grouped images into separate folders
    os.makedirs(out_folder, exist_ok=True)

    # move each image into its cluster folder
    for cluster, images in grouped_images.items():
        cluster_folder = os.path.join(out_folder, f"cluster_{cluster}")
        os.makedirs(cluster_folder, exist_ok=True)
        for img in images:
            img_name = os.path.basename(img)
            os.rename(img, os.path.join(cluster_folder, img_name))

# a different approach for grouping images using cosine similarity
def group_images_with_cosine_similarity(image_folder, model, out_folder=".", n_clusters=5, linkage='ward'):
    # get all PNG files in the folder
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

    # extract features for all images
    features_list = [extract_features(img, model) for img in image_files]
    features_array = np.array(features_list)

    # compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(features_array)

    # perform hierarchical clustering
    # no idea what that means
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusters = clustering.fit_predict(1 - similarity_matrix)

    # group images by cluster
    grouped_images = {}
    for i, cluster in enumerate(clusters):
        if cluster not in grouped_images:
            grouped_images[cluster] = []
        grouped_images[cluster].append(image_files[i])

    # save grouped images into separate folders
    os.makedirs(out_folder, exist_ok=True)

    # move each image into its cluster folder
    for cluster, images in grouped_images.items():
        cluster_folder = os.path.join(out_folder, f"cluster_{cluster}")
        os.makedirs(cluster_folder, exist_ok=True)
        for img in images:
            img_name = os.path.basename(img)
            os.rename(img, os.path.join(cluster_folder, img_name))
