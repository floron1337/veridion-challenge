# Website Logo Matching and Grouping
## Overview
This project tackles the challenge of matching and grouping websites based on the similarity of their logos. In today’s competitive landscape, logos play a vital role in a company's identity. Your brand’s logo is more than just a graphic—it’s a symbol that, when recognized, instantly connects a customer with the company’s services and values. Our solution aims to automatically group similar logos, making it easier to identify relationships between websites and their brands.
## Problem Statement
Given a list of websites and their associated logos, the goal is to:
- **Extract logos** from the provided dataset.
- **Pre-process** the logos by cleaning and removing extraneous elements (such as backgrounds).
- **Cluster or group** the logos based on visual similarity so that logos representing the same or similar brands are placed together.

The final output is expected to be a set of groups where each group contains one or more websites. Some logos might be unique, standing on their own, whereas others will cluster together based on their similarity.
## Approach
### 1. Data Extraction and Preprocessing
- **Data Ingestion:**
The dataset is provided in the form of a parquet file containing website information and logo metadata. The dataset is read using the pandas library.
- **Logo Download:**
A dedicated function is used to download all logos based on the provided URLs and store them in a specified directory.
- **Background Removal:**
Before processing for similarity, logos undergo a preprocessing step to remove unnecessary backgrounds. This is key to ensure the core logo features are maintained, allowing for more accurate similarity comparisons.

### 2. Feature Extraction Using Pretrained Models
- **Pretrained CNN Models:**
To extract meaningful visual features from each logo, the solution leverages pretrained convolutional neural network models (e.g., VGG16, ResNet50). Such models are well-suited for identifying patterns and distinguishing visual features in images.
- **Feature Pooling:**
By setting up the model to use pooling techniques (global average pooling in this case), the solution condenses the image representation into a fixed-length feature vector suitable for similarity comparison.

### 3. Matching and Grouping
There are two primary methods implemented for grouping:
- **Cosine Similarity:**
    - The primary method leverages cosine similarity to compare the extracted feature vectors from each logo.
    - A similarity threshold is set to decide if two logos belong to the same group. This approach is intuitive and does not rely on traditional ML clustering algorithms like DBSCAN or k-means.

- **Clustering Alternative:**
    - An alternative method using clustering (with a predefined number of clusters) is available for experimentation and comparison.
    - This offers another angle to group logos but may require tuning parameters such as the number of clusters.

### 4. Output
- **Group Formation:**
The program outputs multiple groups, each containing one or more website logos. This segmentation helps in quickly identifying which websites share a similar branding style.
- **Coverage and Accuracy:**
The intended goal is to extract logos for more than 97% of websites in the dataset. A robust grouping method will ensure that logos which are visually similar—even with some variation—get paired correctly.

## Considerations and Next Steps
- **Extending the Approach:**
Experimenting with various similarity thresholds, exploring additional image features, or combining multiple methods could further refine group accuracy. Also, the storage and processing mechanisms can be improved to reduce the time and space complexity of the solution.
- **Modularity:**
While the current solution is built for a manageable dataset size, the underlying approach and methodology are chosen with scalability and modularity in mind, making it easy to improve the current codebase without breaking pre-existing software built on top of it.
- **Evaluation:**
The final solution should be evaluated not only on quantitative metrics, such as grouping accuracy, but also on qualitative assessments by human reviewers to ensure the grouped logos are intuitively similar.

## Thought Process and Difficulties

Developing this solution involved a careful balance between achieving accuracy in logo similarity and ensuring the overall system remains both robust and scalable. Here are some insights into the thought process and challenges encountered during development:

- **Obtaining the logo:**  
  Initially, I attempted to scrape the web for each logo, but the results were inconsistent and of low quality. Although I could have used an upscaler to improve the images, I discovered a simpler and more time-efficient solution: using a free online API to download each company's logo, which provided good accuracy. Please DM me if you'd like me to develop my own web scraper for this project.

- **Feature Extraction Complexity:**  
  I needed a reliable metric to group images, so I used pretrained CNN models such as VGG16 or ResNet50 to efficiently extract visual features. However, tuning the model required several experiments to balance detail and abstraction appropriately. Ultimately, VGG16 was chosen through trial and error.

- **Background Removal:**  
  The varied and noisy backgrounds in the logo images necessitated a reliable background removal process. Establishing an effective cleanup workflow was particularly challenging due to differences in image quality and design, which are critical to ensure that only the essential features contribute to the similarity comparison. I also found it curious that background removal is more resource-intensive in terms of hardware and time than feature extraction and grouping, suggesting there is room for improvement in this area.

- **Choice of Similarity Metrics:**  
  Two primary methods were explored: cosine similarity and clustering. Determining an optimal similarity threshold for cosine similarity was non-trivial; it required iterative testing to find a balance that accurately groups logos without over-clustering dissimilar images. One advantage of cosine similarity is that it allows the program to determine the necessary number of groups dynamically, rather than restricting it to a fixed number of clusters.

## Conclusion
This project demonstrates a comprehensive approach to matching and grouping websites by the similarity of their logos. By combining effective preprocessing, deep feature extraction, and simple yet powerful similarity metrics, the solution provides a robust foundation to identify related brands based on visual identity. The creative flexibility in using various tools and methods positions this approach as a promising solution for scalable real-world applications.
