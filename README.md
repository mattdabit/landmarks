# Landmarks

A repository focused on landmark recognition and retrieval.

## Repository Structure

The repository is organized into several key directories:

- `data/`: Contains all dataset files and extracted features
- `images/`: Contains all images for EDA
- `data_extraction.ipynb`: Notebook for processing and extracting features from images. [Link](https://github.com/mattdabit/landmarks/blob/main/data_extraction.ipynb)
- `eda.ipynb`: Notebook for analyzing features. [Link](https://github.com/mattdabit/landmarks/blob/main/eda.ipynb)

## Data Extraction Process

The data extraction code processes the landmark images to create parquet files containing extracted features:

## Important Note

The joined_features_all.parquet file, which contains combined features for all landmarks, is not included in the repository
due to its large size. This file can be generated locally using the data extraction notebook, however this process can take several days.
    