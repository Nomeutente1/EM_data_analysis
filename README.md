# Description
This project provides a useful tool for analyzing electron microscopy data. 
More specifically, the `morph_analysis.py` module offers various functions that iteratively analyze several morphological properties of networks (such as porosity, tortuosity, fractal dimension, etc.) obtained from in-situ measurement movies. This module is highly valuable as it facilitates the investigation of how specific properties evolve over time, thereby reducing the time required for data analysis.

# Features
- Normalization for a better image contrast (optional)
- Scalebar calibration 
- ROI selection 
- Plotting a specific property value as a function of time
- Generating a .csv file for a subsequent analysis with other programs (such as curve fitting)
- Provides images and corresponding tables for the segmentation measurements

# Installation
'EM_data_analysis' requires Python 3.6 or above. 

## Installation using 'pip' (recommended):
'''bash
pip install EM_data_analysis
  
# Usage
Create an empty directory (will be used to store all the frames created from the movie). Then, simply input the path of the movie, the directory created to store the frames, frame rate and the time length.

'''python
from EM_data_analysis import morph_analysis as ma

Network_analysis("path-of-the-movie", "path-of-the-directory", frame-rate, time-length) '''

The movie format must be explicit and could be in AVI or MP4 format.

# Contributions
EM_data_analysis is created by Mattia Lizzano





