# Landmarks

A repository focused on landmark recognition and retrieval.

## Repository Structure

The repository is organized into several key directories:

- `data/`: Contains all dataset files and extracted features
- `images/`: Contains all images for EDA
- `data_extraction.ipynb`: Notebook for processing and extracting features from
  images. [Link](https://github.com/mattdabit/landmarks/blob/main/data_extraction.ipynb)
- `eda.ipynb`: Notebook for analyzing features. [Link](https://github.com/mattdabit/landmarks/blob/main/eda.ipynb)

## Local Installation

1. Clone the repository
2. Use a python environment manager. I prefer conda.
3. Create and activate conda environment
    ```
    conda env create -f environment.yml   
    conda activate landmark
    ```

## About the Dataset

The dataset was procured from
the [Google Landmark Retrieval 2021
](https://www.kaggle.com/competitions/landmark-retrieval-2021/data) kaggle competition.
The dataset contains over 1.5 million images and is over 100GB of data.
The purpose of the competition is to develop a model that can efficiently fetch landmark images.

## Why is this important? (Business Understanding)

Image similarity, detection and retrieval are an important problem in computer vision. If you have a program that can
decipher what it is looking at, then other possibilities will open up. In this project I am working on landmark
recognition, and the same models I create for this project can be applied to a variety of problems. For this project,
imagine an app that gives recommendations to nearby landmarks based on the photos you upload during your vacation. In a
different context, imagine a clothing company finding that shirt you saw on the street but could not find in a store.
This would be helpful for a clothing company trying to attract customers. If a machine can detect the similarity, then
we
can even use it for security ala facial recognition. The possibilities with image recognition are endless,
without image recognition you can say goodbye to facial recognition, fast toll booths that photograph your license
plate, early wildfire detection with satellite images and much more.

## Data Extraction Process

The data extraction code processes the landmark images to create parquet files containing extracted features:

1. Image Stats - Basic details of the image
    1. Height
    2. Width
    3. Aspect Ratio
    4. Mean RGB
2. Embeddings
    1. An Embedding of an image is a numeric vector that captures the content of the image (shape, textures, objects)
        1. I utilized a pre-trained model [ResNet-50](https://medium.com/@f.a.reid/image-similarity-using-feature-embeddings-357dc01514f8) from the torchvision package.
    2. These were quite large and could not be joined to the main dataset. I still will like to leverage these features,
       as I think the dimensionality reduction I do may have lost too much data.
3. Embeddings reduced to 2 Dimensions
    1. After noticing the size of the embedding files I knew I had to reduce the amount of data I had for training.
    2. I utilized a technique called t-distributed stochastic neighbor embedding, which reduces data to two points.
    3. As we will see in the EDA, the values produced
       by [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) distinguished models into
       clusters.
4. Local Binary Pattern
    1. [LBP](https://en.wikipedia.org/wiki/Local_binary_patterns) is a texture descriptor. It does this by comparing a pixel to its neighboring pixels. It captures the
       intensity
       of each pixel and compares their intensities. It will assign 1 if the neighboring pixel's intensity is greater
       than the pixel currently being assessed. Otherwise, LBP will assign 0. The algorithm then combines the binary
       values of all the neighboring pixels to create a value for the pixel being assessed. It does this for all the
       pixels in the image to create a binary code representing the texture of the image.

I chose to create parquets over csvs for space and speed concerns.
Next, I joined all the parquets into 1 for each feature type (excluding embedding). 
Finally, I joined all that data into the train csv and produced a new dataset to train on with the features above. 

⚠️IMPORTANT NOTE: The joined_features_all.parquet file, which contains combined features for all landmarks, is not
included in the
repository
due to its large size. This file can be generated locally using the data extraction notebook, however this process can
take several days.
    