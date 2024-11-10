# Unsupervised_Brain_Tissue_Segmentation_with_GMM_and_EM

This project implements a medical imaging pipeline to perform clustering and segmentation on MRI images using a combination of k-means clustering and the Expectation-Maximization (EM) algorithm. The segmentation is evaluated using Dice similarity scores.

## Project Structure

- **`data_loader.py`**: Functions for loading and preprocessing MRI images.
- **`clustering.py`**: Clustering algorithms, including k-means initialization and EM.
- **`evaluation.py`**: Functions to calculate Dice similarity scores for segmentation.
- **`main.py`**: Main script that runs the pipeline from loading data to evaluation.

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
