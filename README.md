
# Cartoonify Project

## Author
Shimon Ifrach

## Overview
This project contains code for converting images into cartoon-like representations. It was developed as part of an assignment for the Introduction to Computer Science course (Exercise 6, 2022).

## Files
1. **cartoonify.py**: This is the main script that performs the cartoonification of images. It imports necessary functions from the helper script and contains several functions to process images.
2. **ex6_helper.py**: This helper script provides supporting functions for image processing. It was provided by the course staff for use in Exercise 6.

## Description of `cartoonify.py`
- **separate_channels(image)**: Separates the RGB channels of the input image.
- **combine_channels(red_channel, green_channel, blue_channel)**: Combines the separate RGB channels back into an image.
- **convert_to_grayscale(image)**: Converts the input image to grayscale.
- **blur_image(image, kernel_size)**: Applies a Gaussian blur to the input image.
- **detect_edges(image, threshold)**: Detects edges in the input image using the Sobel operator.
- **cartoonify(image_path, output_path, blur_kernel_size=5, edge_threshold=100)**: The main function that reads an image, processes it to create a cartoon effect, and saves the result.

## Usage
1. Place the images you want to cartoonify in the same directory as the scripts.
2. Run `cartoonify.py` with the path to the input image and the desired output path.
   ```bash
   python cartoonify.py input_image.jpg output_image.jpg
   ```
3. Optional parameters include `blur_kernel_size` and `edge_threshold` to adjust the cartoonification process.

## Dependencies
- Python 3.x
- Required libraries: `math`, `sys`

