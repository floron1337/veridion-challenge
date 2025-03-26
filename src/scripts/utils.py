import constants
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keras.src.applications.vgg16 import preprocess_input
from keras.src.utils import img_to_array
import requests
from PIL import Image
import rembg

# obtain the logo of a domain, using Clearbit's Logo API
# the logo is saved at [dest] with the name [domain].PNG
# optionally, you can pass out_file as a parameter to specify the output file's name
def download_logo(domain, dest='.', output_file_name=None):
    response = requests.get(url=f'{constants.CLEARBIT_LOGO_API_URL}/{domain}')
    if response.status_code != 200:
        return None

    os.makedirs(dest, exist_ok=True)
    out_file_name = f'{domain}.png' if output_file_name is None else f'{output_file_name}.png'
    out_file = os.path.join(dest, out_file_name)

    with open(out_file, "wb") as file:
        file.write(response.content)

    return f'{dest}/{out_file_name}'


def download_all_logos(df, out_dest='.'):
    for row in df.itertuples():
        domain = row[1]

        logo_file = download_logo(domain, dest=out_dest)
        if logo_file is None:
            print(f"Error downloading logo for domain {domain}")
            continue


def remove_image_logo(file_path):
    initial_image = Image.open(file_path)
    image_array = np.array(initial_image)
    output_array = rembg.remove(image_array)
    output_image = Image.fromarray(output_array)
    output_image.save(file_path)


def remove_all_backgrounds(dir_path):
    for i, file in enumerate(os.listdir(dir_path)):
        if file.endswith(".png"):
            file_path = os.path.join(dir_path, file)
            remove_image_logo(file_path)
            print(f"file {i}: background removal successful")


class ImageGrouper:
    def __init__(self, model):
        self.__model = model

    # added a safer way to change models
    def change_model(self, model):
        self.__model = model

    # obtain useful information from an image using a pretrained model
    def __extract_features(self, img_path):
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_data = img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = self.__model.predict(img_data)
        return features.flatten()

    # KMeans clustering for grouping images
    def group_images_with_clustering(self, image_folder, out_folder=".", n_clusters=5, random_state=42):
        # get all PNG files in the folder
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

        # extract features for all images
        features_list = [self.__extract_features(img) for img in image_files]
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


        print(f"done. output folder: {out_folder}")


    # a different approach for grouping images using cosine similarity
    def group_images_with_cosine_similarity(self, image_folder, out_folder=".", similarity_threshold=0.8):
        # get all PNG files in the folder
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

        # extract features for all images
        features_list = [self.__extract_features(img) for img in image_files]
        features_array = np.array(features_list)

        # compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(features_array)

        grouped_images = []
        used_indices = set()

        for i in range(len(image_files)):
            if i in used_indices:
                continue
            used_indices.add(i)
            group = [image_files[i]]
            for j in range(i + 1, len(image_files)):
                if j not in used_indices and similarity_matrix[i, j] >= similarity_threshold:
                    group.append(image_files[j])
                    used_indices.add(j)
            grouped_images.append(group)

        # out_folder to store grouped images
        os.makedirs(out_folder, exist_ok=True)

        for group_id, group in enumerate(grouped_images):
            group_folder = os.path.join(out_folder, f"group_{group_id}")
            os.makedirs(group_folder, exist_ok=True)
            for img in group:
                img_name = os.path.basename(img)
                os.rename(img, os.path.join(group_folder, img_name))

        print(f"done. output folder: {out_folder}")
