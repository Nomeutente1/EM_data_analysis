This project aims to provide a useful tool for analysing elecron microscopy data. 
More specifically, "morph_analysis" iteratively analyses several morphological properties of networks (such as porosity, fractal dimension, tortuosity, ecc.) obtained from in-situ measurements.

This module is highly valuable as it allows for easy investigation on how specific properties evolve over time, thereby reducing the time required for data analysis. 
The only parameters needed are a movie file (in AVI or MP4 format) and information regarding the time duration and frame rate.

How to use:
use the function "DyNetAn_project" as follows:

DyNetAn_project("filepath_of_the_movie", "filepath_of_a_temporary_directory", frame_rate, time_duration)
