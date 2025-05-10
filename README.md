# K-Means Image Compression

This project demonstrates how to compress images using the K-Means clustering algorithm. The goal is to reduce the number of unique colors in the image while retaining visual similarity.

## Features
- Load and process images.
- Apply K-Means clustering for color reduction.
- Save and compare the compressed image with the original.

## How It Works
1. The image is read and its pixels are normalized to the [0, 1] range.
2. Pixels are flattened into a 2D array.
3. K-Means clustering groups similar colors, reducing the color palette.
4. The compressed image is reconstructed using the cluster centroids.


## Streamlit Application 
https://k-means-image-compression.streamlit.app/

