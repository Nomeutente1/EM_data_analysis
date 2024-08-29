import cv2
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import openpnm as op
import os.path
from moviepy.editor import VideoFileClip
import os
import shutil
import scipy.ndimage as sn
from IPython.display import clear_output
from skimage.morphology import skeletonize
import logging
from porespy.filters import trim_nonpercolating_paths
from porespy.generators import faces
from porespy.tools import Results
import pandas as pd
import dataframe_image as dfi
#
#
#
#
def choose_threshold(img, thr, image_rgb):
    
    """
    Interactive function to choose an optimal threshold value for an image.
    
    Parameters:
    - img: Grayscale image array.
    - initial_thr: Initial threshold value (default: 0).
    - image_rgb: RGB representation of the original image.
    
    Returns:
    - Optimal threshold value chosen interactively.
    """
    if thr == 0:
        Thr_not_ok = True
        
        while Thr_not_ok:
            
            # Apply thresholding
            clear_output(wait=True)
            thr = int(input("Insert threshold value (between 0-255): "))
            clear_output(wait=True)
            
            ret, bw_img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

            # Invert the binary image
            inverted_img = cv2.bitwise_not(bw_img)
            
            # Create a mask based on the thresholded image
            mask = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)

            # Generate a colored mask 
            red_colour = (0, 0, 255)
            colored_mask = np.zeros_like(inverted_img)
            colored_mask[mask == 255] = red_colour
            
            # Overlay the original image with the colored mask
            transparency = 0.6
            overlay = cv2.addWeighted(image_rgb, transparency, colored_mask, 1 - transparency, 0) 
            
            # Display the overlay
            cv2.imshow("BINARY IMAGE, PRESS ENTER TO CONTINUE", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Ask the user if the threshold is okay
            how_thr = input("Is threshold ok? Answer y/n: ").lower()
            clear_output(wait=True)
            
            if how_thr == 'n':
               continue
            else:
                break
            
        return thr
########################################################################################################################################################
                                            
########################################################################################################################################################
def click_coordinates(image):
    """
    Extract the coordinates of two points from an image via mouse clicks.
    
    Parameters:
    - image: A copy of the original image in RGB format.
    
    Returns:
    - A tuple containing the coordinates of the two selected points.
    """ 
    # Copy of the image for manipulation
    image_copy = image.copy()
    
    # List to store the coordinates of the clicked points
    clicked_points = []
    
    # Define the mouse click event handler
    def click_event(event, x, y, flags, params):
        nonlocal clicked_points
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Store the coordinates of the left mouse button down event
            clicked_points.append((x, y))
            
            # Draw a small circle at the clicked point
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
            
            # Update the displayed image
            cv2.imshow("CLICK OVER BOTH ENDS OF THE SCALEBAR, PRESS ENTER TO CONTINUE", image_copy)
            
            # Check if two points have been selected
            if len(clicked_points) == 2:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return clicked_points # Return the coordinates upon selection completion
    
    
    # Set up the mouse callback
    cv2.imshow("CLICK OVER BOTH ENDS OF THE SCALEBAR, PRESS ENTER TO CONTINUE", image_copy)
    cv2.setMouseCallback("CLICK OVER BOTH ENDS OF THE SCALEBAR, PRESS ENTER TO CONTINUE", click_event)
    
    # Wait for user interaction
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return clicked_points
########################################################################################################################################################
                                            
########################################################################################################################################################

def binary_image_processing(filepath, thr, roi):
    """
    Process an image for binaryization, including optional erosion/dilation adjustments.
    
    Parameters:
    - filepath: Path to the image file.
    - thr: Threshold value for binaryization. Initially 0, updated by `choose_threshold`.
    - roi: Region of interest defined by top-left and bottom-right coordinates.
    
    Returns:
    - greyscale_img: Greyscale version of the ROI.
    - img_array: Binaryized image array with foreground pixels set to 1 and background to 0.
    - thr: Final threshold value used.
    """
    global norm, norm_roi
    # Read the image and crop to ROI
    img_rgb = cv2.imread(filepath)
    new_img_rgb = img_rgb.copy()
    greyscale_img_rgb = img_rgb.copy()[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]    
    greyscale_img = cv2.cvtColor(greyscale_img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Convert full image to greyscale and crop to ROI
    img_rgb = img_rgb[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    flag_erode = 0
    flag_dilate = 0
    
    if norm == 'n':
        # If the threshold value is 0 (should be 0 only for the first iteraction) the function allows to select it manually.
        if thr == 0:
            thr = choose_threshold(img_rgb, thr, greyscale_img_rgb)
            
            # Perform binaryization
            ret, bw_img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
            inverted_img = cv2.bitwise_not(bw_img)
            img_array = np.array(inverted_img)
        
            # Removing the noise
            img_array = cv2.erode(img_array, None, iterations = 1)
            img_array = cv2.dilate(img_array, None, iterations = 1)
            
            # Update the image without noise
            cv2.imshow("BINARY IMAGE AFTER ERODE/DILATE, PRESS ENTER TO CONTINUE", img_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Optional erosion and dilation
            mod = True
            while mod:
                ask_modify = int(input("Press: 1 to erode, 2 to dilate, 3 for pass: \n"))
                clear_output(wait=True)
                if ask_modify == 1:
                    img_array = cv2.erode(img_array, None, iterations = 1)
                    flag_erode += 1
                    cv2.imshow("BINARY IMAGE AFTER ERODE, PRESS ENTER TO CONTINUE", img_array)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                elif ask_modify == 2:
                    img_array = cv2.dilate(img_array, None, iterations = 1)
                    flag_dilate += 1
                    cv2.imshow("BINARY IMAGE AFTER DILATE, PRESS ENTER TO CONTINUE", img_array)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                elif ask_modify == 3:
                    mod = False
        
            # Convert binary image array to 0 and 1
            img_array [img_array == 255] = 1
            
        else:
            # Perform binaryization
            ret, bw_img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
            inverted_img = cv2.bitwise_not(bw_img)
            img_array = np.array(inverted_img)
            img_array = cv2.erode(img_array, None, iterations = flag_erode + 1)
            img_array = cv2.dilate(img_array, None, iterations = flag_dilate + 1)
            
            # Convert binary image array to 0 and 1
            img_array [img_array == 255] = 1
            
    elif norm == 'y':
        img_norm = new_img_rgb[norm_roi[0][1]:norm_roi[1][1], norm_roi[0][0]:norm_roi[1][0]] 
        
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
        mean, STD  = cv2.meanStdDev(img_norm)
        offset = 0.2
        
        clipped = np.clip(img, mean - offset*STD, mean + offset*STD).astype(np.uint8)
        img_normalized = cv2.normalize(clipped, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        img_normalized_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
        
        
        # If the threshold value is 0 (should be 0 only for the first iteraction) the function allows to select it manually.
        if thr == 0:
            thr = choose_threshold(img_normalized_rgb, thr, greyscale_img_rgb)
            
            # Perform binaryization
            ret, bw_img = cv2.threshold(img_normalized, thr, 255, cv2.THRESH_BINARY)
            inverted_img = cv2.bitwise_not(bw_img)
            img_array = np.array(inverted_img)
        
            # Removing the noise
            img_array = cv2.erode(img_array, None, iterations = 1)
            img_array = cv2.dilate(img_array, None, iterations = 1)
            
            # Update the image without noise
            cv2.imshow("BINARY IMAGE AFTER ERODE/DILATE, PRESS ENTER TO CONTINUE", img_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Optional erosion and dilation
            mod = True
            while mod:
                ask_modify = int(input("Press: 1 to erode, 2 to dilate, 3 for pass: \n"))
                clear_output(wait=True)
                if ask_modify == 1:
                    img_array = cv2.erode(img_array, None, iterations = 1)
                    flag_erode += 1
                    cv2.imshow("BINARY IMAGE AFTER ERODE, PRESS ENTER TO CONTINUE", img_array)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                elif ask_modify == 2:
                    img_array = cv2.dilate(img_array, None, iterations = 1)
                    flag_dilate += 1
                    cv2.imshow("BINARY IMAGE AFTER DILATE, PRESS ENTER TO CONTINUE", img_array)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                elif ask_modify == 3:
                    mod = False
        
            # Convert binary image array to 0 and 1
            img_array [img_array == 255] = 1
            
        else:
            ret, bw_img = cv2.threshold(img_normalized, thr, 255, cv2.THRESH_BINARY)
            inverted_img = cv2.bitwise_not(bw_img)
            img_array = np.array(inverted_img)
            img_array = cv2.erode(img_array, None, iterations = flag_erode + 1)
            img_array = cv2.dilate(img_array, None, iterations = flag_dilate + 1)
            img_array [img_array == 255] = 1
            
    return greyscale_img, img_array, thr
########################################################################################################################################################
                                            
########################################################################################################################################################

def calculate_rescaling_factor(filepath):
    """
    Calculate the rescaling factor of an image based on the distance between two points.
    
    Parameters:
    - filepath: Filepath of the image to analyze.
    
    Returns:
    - rescaling_factor: The rescaling factor calculated from the real distance and the distance derived from the coordinates.
    - mag_ord: Order of magnitude (micrometer or nanometer) based on user input.
    """
    # Load the image
    image = cv2.imread(filepath)

    # Extract coordinates of two points
    coordinates = click_coordinates(image)
    line_length = np.sqrt((coordinates[1][0] - coordinates[0][0])**2)
    
    # Prompt user for scalebar length and order of magnitude
    scalebar = float(input("Insert scalebar length in real units: "))
    mag_ord = None
    flag = int(input("Select order of magnitude:\n 1- micrometer\n 2- nanometer \n"))
    clear_output(wait=True)
    
    # Assign order of magnitude based on user choice
    if flag == 1:
        mag_ord = "um"
    if flag == 2:
        mag_ord = "nm"

    rescaling_factor = scalebar / line_length

    return rescaling_factor, mag_ord
########################################################################################################################################################
                                            
########################################################################################################################################################

def ROI_selection_and_cropping(filepath):
    """
    Crops an image.

    Parameters:
    - filepath (str): Path to the image file to be cropped.

    Returns:
    - cropped_image: The cropped portion of the image corresponding to the ROI.
    - refPt: ROI coordinates
    """
    # Load the image
    image = cv2.imread(filepath)
    
    # Initialize global variables
    refPt = []
    cropping = False
    
    # Mouse callback function
    def click_and_crop(event, x, y, flags, param):
        nonlocal refPt , cropping 
    
        # If the left mouse button was clicked, record the (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True
    
        # Check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # Record the ending (x, y) coordinates and indicate that the cropping operation is finished
            refPt.append((x, y))
            
            cropping = False
    
            # Draw a rectangle around the region of interest
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("DRAW A RECTANGLE FROM UP-LEFT TO BOTTOM-RIGHT, R TO RESET, C TO CONTINUE", image)
    
    # Create a window and bind the mouse callback function to it
    cv2.namedWindow("DRAW A RECTANGLE FROM UP-LEFT TO BOTTOM-RIGHT, R TO RESET, C TO CONTINUE")
    cv2.setMouseCallback("DRAW A RECTANGLE FROM UP-LEFT TO BOTTOM-RIGHT, R TO RESET, C TO CONTINUE", click_and_crop)
    
    # Show the image and wait for the user to finish selecting the ROI
    while True:
        cv2.imshow("DRAW A RECTANGLE FROM UP-LEFT TO BOTTOM-RIGHT, R TO RESET, C TO CONTINUE", image)
        key = cv2.waitKey(1) & 0xFF
        
        # If the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = cv2.imread(filepath)
            refPt = []
            cropping = False
    
        # If the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    
    cv2.destroyAllWindows()
    
    # Perform the cropping operation
    if len(refPt) == 2:
        cropped_image = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("CROPPED IMAGE, PRESS ENTER TO CONTINUE", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("not.")

    return cropped_image, refPt
########################################################################################################################################################
                                            
########################################################################################################################################################

def fractal_dimension(image):
    """
    Calculates the fractal dimension for different box edge lengths in a binary image.

    Parameters:
    - image_binary (numpy.ndarray): Binary representation of the image.

    Returns:
    - data.slope: array containing the fractal dimension values up to the 7th element.
    - data.size: array containing the box lengths values up to the 7th element.
    """          
    data = ps.metrics.boxcount(image)

    # Data.slope are the FD values before the finite-size divergence, 
    # data.size are the box edge length before the finite-size divergence.
    return data.slope[0:7], data.size[0:7]
########################################################################################################################################################
                                            
########################################################################################################################################################

def porosity(image):
    """
    Calculates the porosity of a binary image, defined as the ratio of the area covered by the background to the total area.

    Parameters:
    - image_binary (numpy.ndarray): Binary representation of the image, where the foreground is represented by 1s and the background by 0s.

    Returns:
    - float: The calculated porosity value, ranging from 0 to 1, where 0 indicates no porosity (all area covered by the foreground) 
             and 1 indicates full porosity (no area covered by the foreground).
    """
    initial_porosity = ps.metrics.porosity(image)
    corrected_porosity = 1-initial_porosity
    return corrected_porosity
########################################################################################################################################################
                                            
########################################################################################################################################################

def connected_areas(image):
    """
    Calculates the ratio of connected areas within a binary image.

    Parameters:
    - image_binary (numpy.ndarray): Binary representation of the image, where the foreground is represented by 1s and the background by 0s.

    Returns:
    - float: The ratio of connected areas within the image, calculated as the sum of connected pixels divided by the total number of pixels.
    """

    # Initialize arrays for inlets and outlets along the edges of the image.
    inlets = np.zeros_like(image)
    inlets[0, :] = True # Western edge
    outlets = np.zeros_like(image)
    outlets[-1, :] = True # Eastern edge

    full_image = np.zeros_like(image)
    
    # In order: Western edge, Eastern edge, Nothern edge, Southern edge
    indices = [(0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1)]
    
    # Cycle all over the edge, trimming the non connected area in that particular configuration. Then, all the connected area are stored in full_image
    for index, (i, j) in enumerate(indices):
        inlets[i, j] = True
        outlets[i, j] = True
        current_img= ps.filters.trim_nonpercolating_paths(im=image, inlets=inlets, outlets=outlets)
        full_image[current_img == True] = 1
        
    # Ratio between conncected and total area
    connected_area_ratio = (np.sum(full_image == 1)/np.sum(image == 1))
        
    return connected_area_ratio
########################################################################################################################################################
                                            
########################################################################################################################################################
def tortuosity_fd(im, axis = 1, solver=None):
    '''
    Calculates the tortuosity value between the left and right edge of the image.

    Parameters:
    ----------
    - im (numpy.ndarray): Binary representation of the image, where the foreground is represented by 1s and the background by 0s.
    - axis : optional, the default is 1.
    - solver : optional the default is None.

    Returns:
    - data.tortuosity: tortuosity value
    '''
    
    logger = logging.getLogger(__name__)
    ws = op.Workspace()
   
    if axis > (im.ndim - 1):
        raise Exception(f"\'axis\' must be <= {im.ndim}")

    openpnm_v3 = op.__version__.startswith("3")

    # Obtain original porosity
    eps0 = im.sum(dtype=np.int64) / im.size

    # Remove floating pores
    inlets = faces(im.shape, inlet=axis)
    outlets = faces(im.shape, outlet=axis)
    im = trim_nonpercolating_paths(im, inlets=inlets, outlets=outlets)

    # Check if porosity is changed after trimming floating pores
    eps = im.sum(dtype=np.int64) / im.size
    if not eps:
        logger.warning("No pores remain after trimming floating pores")
        result = Results()
        result.im = im
        result.tortuosity = 0
        result.formation_factor = 0
        result.original_porosity = 0
        result.effective_porosity = 0
    else:    
        if eps < eps0:  # pragma: no cover
            logger.warning("Found non-percolating regions, were filled to percolate")
    
        # Generate a Cubic network to be used as an orthogonal grid
        net = op.network.CubicTemplate(template=im, spacing=1.0)
        if openpnm_v3:
            phase = op.phase.Phase(network=net)
        else:
            phase = op.phases.GenericPhase(network=net)
        phase["throat.diffusive_conductance"] = 1.0
    
        # Run Fickian Diffusion on the image
        fd = op.algorithms.FickianDiffusion(network=net, phase=phase)
    
        # Choose axis of concentration gradient
        inlets = net.coords[:, axis] <= 1
        outlets = net.coords[:, axis] >= im.shape[axis] - 1
    
        # Boundary conditions on concentration
        cL, cR = 1.0, 0.0
        fd.set_value_BC(pores=inlets, values=cL)
        fd.set_value_BC(pores=outlets, values=cR)
    
        if openpnm_v3:
            if solver is None:
                solver = op.solvers.PyamgRugeStubenSolver(tol=1e-8)
            fd._update_A_and_b()
            fd.x, info = solver.solve(fd.A.tocsr(), fd.b)
            if info:
                raise Exception(f"Solver failed to converge, exit code: {info}")
        else:
            fd.settings.update({"solver_family": "scipy", "solver_type": "cg"})
            fd.run()
    
        # Calculate molar flow rate, effective diffusivity and tortuosity
        r_in = fd.rate(pores=inlets)[0]
        r_out = fd.rate(pores=outlets)[0]
        if not np.allclose(-r_out, r_in, rtol=1e-4):  # pragma: no cover
            logger.error(f"Inlet/outlet rates don\'t match: {r_in:.4e} vs. {r_out:.4e}")
    
        dC = cL - cR
        L = im.shape[axis]
        A = np.prod(im.shape) / L
        Deff = r_in * (L - 1) / A / dC
        tau = eps / Deff
    
        # Attach useful parameters to Results object
        result = Results()
        result.im = im
        result.tortuosity = tau
        result.formation_factor = 1 / Deff
        result.original_porosity = eps0
        result.effective_porosity = eps
    
        conc = np.zeros(im.size, dtype=float)
        conc[net["pore.template_indices"]] = fd["pore.concentration"]
        result.concentration = conc.reshape(im.shape)
        result.sys = fd.A, fd.b
    
        # Free memory
        ws.close_project(net.project)

    return result.tortuosity
########################################################################################################################################################
                                            
########################################################################################################################################################

def analyze_frame_sequence(directory, frame_name_prefix, time_length):
    """
    Analyze a sequence of frames in a directory, performing various analyses based on the `ask` parameter.
    
    Parameters:
    - directory: Directory containing the frame images.
    - frame_name_prefix: Prefix of the frame filenames to match.
    - time_length: Duration of the movie
    
    Returns:
    Depending on the analysis performed, returns various metrics such as fractal dimensions, tortuosities, etc.
    """
    global equidistant_files, saving_folder, norm, norm_roi, ask
    
    # Initialize lists to store analysis results
    frc_dim = []
    box_counts = []
    tortuosity = []
    prsty = []
    conn_mat = []
    isolated_clusters = []
    num_necks = []
    connected_regions = []
    
    # Initializing parameters
    flag = 0
    threshold = 0
    image_binaryzed = np.empty((0,0))
    roi = np.empty((0,0))
  
    
    files = len([entry for entry in os.listdir(directory) if os.path.isfile(os.path.join(directory, entry))])
    equidistant_files = np.round(np.linspace(0, files-1, 6)).astype(int)
    t_step = round(time_length/files, 2)
    
    interaction = 0
    
    # Iterate over files in the directory
    for filename in os.listdir(directory):
         if filename.startswith(frame_name_prefix):
            file_path = os.path.join(directory, filename)
            
            # Perform initial setup for the first frame
            if flag == 0:                    
                rescaling_factor, mag_order = calculate_rescaling_factor(file_path)
                print("Rescaling factor =", rescaling_factor, "unit: ", mag_order, "/pixel")
                
                if ask == 6:
                    
                    factor = float(input("Insert factor for neck thresholding:"))
                    neck_thr = factor * rescaling_factor
                    print("Neck threshold = ", neck_thr, "Unit: ", mag_order)
            
                
                if norm == 'n':
                    crop, roi = ROI_selection_and_cropping(file_path)
                    gray_img, image_binaryzed, threshold = binary_image_processing(file_path, threshold, roi)
                    flag = 1
                elif norm == 'y':
                    crop, roi = ROI_selection_and_cropping(file_path)
                    norm_roi = ROI_selection_and_cropping(file_path)[1]
                    gray_img, image_binaryzed, threshold = binary_image_processing(file_path, threshold, roi)
                    flag = 1
                    
            # For subsequent frames, reuse the last computed threshold
            else: 
                image_binaryzed = binary_image_processing(file_path, threshold, roi)[1]
                
            
            # Perform analysis based on the `ask` parameter
            if ask == 1:
                clear_output(wait=True)
                print("Currently processing image:", filename)
                dimension, boxes = fractal_dimension(image_binaryzed)
                frc_dim.append(dimension) 
                box_counts.append(boxes)

            elif ask == 2:
                clear_output(wait=True)
                print("Currently processing image:", filename)
                tort = tortuosity_fd(image_binaryzed, axis = 1, solver=None)
                tortuosity.append(tort)

            elif ask == 3:
                clear_output(wait=True)
                print("Currently processing image:", filename)
                por = porosity(image_binaryzed)
                prsty.append(por)

            elif ask == 4:
                clear_output(wait=True)
                print("Currently processing image:", filename)
                connected = connected_areas(image_binaryzed)
                conn_mat.append(connected)
                
            elif ask == 5:
                clear_output(wait=True)
                print("Currently processing image:", filename)
                clusters = count_isolated_clusters(image_binaryzed)
                isolated_clusters.append(clusters)
            
            elif ask == 6:
                clear_output(wait=True)
                print("Currently processing image:", filename)
                number_of_necks = ridge_map(image_binaryzed, rescaling_factor, neck_thr)
                num_necks.append(number_of_necks)
            
            elif ask == 7:
                
                clear_output(wait=True)
                print("Currently processing image:", filename)
                segmentation = count_segmented_layer(image_binaryzed, interaction, equidistant_files, t_step, rescaling_factor)
                interaction += 1
                connected_regions.append(segmentation)
    
    # Return the appropriate analysis results based on the `ask` parameter
    if ask == 1:
        return frc_dim, box_counts[0]
    elif ask == 2:
        return tortuosity
    elif ask == 3:
        return prsty
    elif ask == 4:
        return conn_mat
    elif ask == 5:
        return isolated_clusters
    elif ask == 6:
        return num_necks
    elif ask == 7:
        return connected_regions
########################################################################################################################################################
                                            
########################################################################################################################################################

def plot_analysis_results(directory, frame_name_prefix, time_length):
    """
    Plots various analysis results against time for a sequence of images.

    Parameters:
    - directory (str): Directory path containing the dataset.
    - name (str): Name of each frame in the dataset.
    - time_length (float): Total duration of the analysis in seconds.

    Note: This function assumes the existence of a function named `analyze_frame_sequence` that processes each frame and returns the desired analysis result.
    """
    global ask, saving_folder, norm
    ask = int(input("Select analysis: \n1 - Fractal dimension \n2 - Tortuosity \n3 - Porosity \n4 - Ratio of connected material \n5 - Number of clusters \n6 - Number of necks \n7 - Number of connected regions \n"))
    clear_output(wait=True)
    norm = input("Normalize data? Answer: y/n \n")
    clear_output(wait=True)
    
    # This part is only for saving the results
    results_csv_filepath = input("Insert path for saving .csv results: \n")
    
    if ask == 1:

        fractal_dimension, box_edge_length = analyze_frame_sequence(directory, frame_name_prefix, time_length)
        time_step = time_length/len(fractal_dimension)
        x = range(len(fractal_dimension))
        time = list(time_unit * time_step for time_unit in x)
        edge, Time = np.meshgrid(box_edge_length, time)
        fractal = np.array(fractal_dimension)
        indices = np.round(np.linspace(0, len(fractal)-1, 5)).astype(int)
        
        plt.figure()
        plt.plot(edge[0], fractal[indices[0]], marker='o', linestyle='--', label = "Time = " + str(round(time[indices[0]], 2)), color = 'blue')
        plt.plot(edge[0], fractal[indices[1]], marker='+', linestyle='--', label = "Time = " + str(round(time[indices[1]], 2)), color = 'cyan')
        plt.plot(edge[0], fractal[indices[2]], marker='D', linestyle='--', label = "Time = " + str(round(time[indices[2]], 2)), color = 'lawngreen')
        plt.plot(edge[0], fractal[indices[3]], marker='*', linestyle='--', label = "Time = " + str(round(time[indices[3]], 2)), color = 'orange')
        plt.plot(edge[0], fractal[indices[4]], marker='x', linestyle='--', label = "Time = " + str(round(time[indices[4]], 2)), color = 'red')
        
        plt.legend()
        plt.grid(True)
        plt.xlabel('Box edge length (pixel)')
        plt.ylabel('Fractal dimension')
        plt.title('Fractal dimension')

        plt.show()
       
    elif ask == 2:

        tortuosity = analyze_frame_sequence(directory, frame_name_prefix, time_length)
        time_step = time_length/len(tortuosity)
        x = range(len(tortuosity)+1)
        time = list(time_unit * time_step for time_unit in x[1:])
        plt.figure()
        plt.plot(time, tortuosity, color='blue', label='Tortuosity')
        plt.xlabel('Time (s)')
        plt.ylabel('Tortuosity')
        plt.title('Tortuosity vs time')

        plt.show()

    elif ask == 3:
        porosity = analyze_frame_sequence(directory, frame_name_prefix, time_length)
        time_step = time_length/len(porosity)
        x = range(len(porosity)+1)
        time = list(time_unit * time_step for time_unit in x[1:])
        plt.plot(time, porosity, color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Porosity')
        plt.title('Porosity vs time')

    elif ask == 4:
        cn_mat = analyze_frame_sequence(directory, frame_name_prefix, time_length)
        
        
        time_step = time_length/len(cn_mat)
        x = range(len(cn_mat)+1)

        time = list(time_unit * time_step for time_unit in x[1:])
               
        plt.plot(time, cn_mat, color='blue', label='Connected material')
        plt.xlabel('Time (s)')
        plt.ylabel('Ratio')
        plt.title('Ratio of Connected material vs time')
        
    elif ask == 5:
        isolated_clusters = analyze_frame_sequence(directory, frame_name_prefix, time_length)
        
        time_step = time_length/len(isolated_clusters)
        x = range(len(isolated_clusters)+1)
        time = list(time_unit * time_step for time_unit in x[1:])
        
        plt.figure()
        plt.plot(time, isolated_clusters,marker = 'o', linestyle = '--', color='blue')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('# of clusters')
        plt.title('Number of clusters vs time')
        
    elif ask == 6: 
        neck_number = analyze_frame_sequence(directory, frame_name_prefix, time_length)
        
        time_step = time_length/len(neck_number)
        x = range(len(neck_number)+1)
        time = list(time_unit * time_step for time_unit in x[1:])
        
        #fig, ax1 = plt.subplots()
        
        plt.plot(time, neck_number, color='blue', label='Number of necks')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('# of necks')
        plt.title('Number of necks vs time')
        plt.show()
        
    elif ask == 7: 
        
        saving_folder = input("Insert path for saving images and their corresponding tables: \n")
        
        segmented_regions = analyze_frame_sequence(directory, frame_name_prefix, time_length)
        time_step = time_length/len(segmented_regions)
        x = range(len(segmented_regions)+1)
        time = list(time_unit * time_step for time_unit in x[1:])
        
        plt.plot(time, segmented_regions,marker = 'o', linestyle = '--', label='Segmented_regions')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('# of regions')
        plt.title('Number of connected regions vs time')
        plt.show()
    
    if ask == 1:
        combined_array = np.column_stack((time, edge, fractal))
        filename = os.path.join(results_csv_filepath, 'Fractal dimension.csv')
        np.savetxt(filename, combined_array, delimiter=',', fmt='%.6f')
        
        # Optionally, write headers to the CSV file
        with open(filename, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write("Time (s),Counts,Fractal dimension\n" + content)
            file.truncate()
            
    elif ask == 2:
        combined_array = np.column_stack((time, tortuosity))
        filename = os.path.join(results_csv_filepath, 'Tortuosity.csv')
        np.savetxt(filename, combined_array, delimiter=',', fmt='%.6f')
        # Optionally, write headers to the CSV file
        with open(filename, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write("Time (s),Tortuosity\n" + content)
            file.truncate()
            
    elif ask == 3: 
        combined_array = np.column_stack((time, porosity))
        filename = os.path.join(results_csv_filepath, 'Porosity.csv')
        np.savetxt(filename, combined_array, delimiter=',', fmt='%.6f')
        # Optionally, write headers to the CSV file
        with open(filename, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write("Time (s),Porosity\n" + content)
            file.truncate()
            
    elif ask == 4:
        combined_array = np.column_stack((time, cn_mat))
        filename = os.path.join(results_csv_filepath, 'Connected material.csv')
        np.savetxt(filename, combined_array, delimiter=',', fmt='%.6f')
        # Optionally, write headers to the CSV file
        with open(filename, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write("Time (s),Connected material\n" + content)
            file.truncate()
            
    elif ask == 5:
        combined_array = np.column_stack((time, isolated_clusters))
        filename = os.path.join(results_csv_filepath, 'Numb of cluster.csv')
        np.savetxt(filename, combined_array, delimiter=',', fmt='%.6f')
        # Optionally, write headers to the CSV file
        with open(filename, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write("Time (s),# of clusters\n" + content)
            file.truncate()
            
    elif ask == 6:
        combined_array = np.column_stack((time, neck_number))
        filename = os.path.join(results_csv_filepath, 'Necks.csv')
        np.savetxt(filename, combined_array, delimiter=',', fmt='%.6f')
        # Optionally, write headers to the CSV file
        with open(filename, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write("Time (s),NUmber of necks\n" + content)
            file.truncate()
            
    elif ask == 7:
        
        combined_array = np.column_stack((time, segmented_regions))
        filename = os.path.join(results_csv_filepath, 'Connected regions.csv')
        np.savetxt(filename, combined_array, delimiter=',', fmt='%.6f')
        # Optionally, write headers to the CSV file
        with open(filename, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write("Time (s), # of connected regions \n" + content)
            file.truncate()
########################################################################################################################################################
                                             
########################################################################################################################################################

def Network_analysis(movie_path, temp_dir_path):
    """
   Extracts frames from a movie and stores them in a temporary directory for subsequent analysis.

   Parameters:
   - movie_path (str): Full path to the movie file.
   - temp_dir_path (str): Full path to the temporary directory where extracted frames will be stored.

   After extraction, it calls `plot_analysis_results` function to perform analysis on the extracted frames.
   """ 
    # Ensure the temporary directory exists
    os.makedirs(temp_dir_path, exist_ok=True)
    
    # Extract frames from the movie
    clip = VideoFileClip(movie_path)
    time_length = clip.duration
    fps = clip.fps
    clip.subclip(0, time_length).write_images_sequence(temp_dir_path + 'frame%04d.png', fps=fps)
    clip.close()
    
    # Ask the user if they want to delete the image stack after analysis
    remove = input("After the analysis, want to delete the image stack? Answer: y/n")
    clear_output(wait=True)
    
    # Perform analysis on the extracted frames
    plot_analysis_results(temp_dir_path, "frame", time_length)
    
    
    # Delete the temporary directory if the user chooses to do so
    if remove.lower() == 'y':
        shutil.rmtree(temp_dir_path)
    elif remove.lower() == 'n':
        pass
    
########################################################################################################################################################
                                              
########################################################################################################################################################

def count_isolated_clusters(image):
    """
    Counts the number of isolated clusters in a binary image.

    Parameters:
    - image (numpy.ndarray): Binary representation of the image, where the clusters are represented by 1s and the background by 0s.

    Returns:
    - int: The number of isolated clusters found in the image.
    """

    # Creating inlets and outlets. They work as starting and finishing points for trim_non_percolative_paths
    inlets = np.zeros_like(image)
    inlets[0, :] = True # Western edge
    outlets = np.zeros_like(image)
    outlets[-1, :] = True # Eastern edge

    # Full_image is the completed image, trim_image is the same one but only with the isolated clusters (all trimmed parts)
    full_image = np.zeros_like(image)
    trim_image = np.zeros_like(image)

    # Define indices for iterating over edges
    indices = [(0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1)]

    # Iterate over edges to identify connected components
    for index, (i, j) in enumerate(indices):
        inlets[i, j] = True
        outlets[i, j] = True
        current_img= ps.filters.trim_nonpercolating_paths(im=image, inlets=inlets, outlets=outlets)
        full_image[current_img == True] = 1

    # Identify isolated clusters by subtracting full_image from the original image
    trim_image = image - full_image

    # Find contours of isolated clusters
    contours, hierarchy = cv2.findContours(trim_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out contours that represent single pixels
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1]

    

    return len(valid_contours)
########################################################################################################################################################
                                              
########################################################################################################################################################

def ridge_map(image, rescaling_factor, neck_thr):
    """
    Calculate the number of necks from the ridge map obtained from a binaryized image.
    
    Parameters:
    - image: Binaryized image.
    - rescaling_factor: Factor to scale the Euclidean distances.
    - neck_thr: Definition of neck. Everything under this distance is considered neck
    
    Returns:
    - int: Number of necks in the ridge map.
    """
    
    # Calculating the euclidean distance map
    distances = sn.distance_transform_edt(image, sampling=None, return_distances=True, return_indices=False, distances=None, indices=None)*rescaling_factor 
   
    
    # Generate the skeleton of the image
    skeleton = skeletonize(distances).astype(int)
    skeleton[skeleton!= 0] = 1
    
    # To obtain the ridge map, the procedure overlap both distances and skeleton, then calculates the radius of the circle,
    # centered on a skeleton-pixel, to the nearest background point
    
    # Create an empty map for the minimum radius
    min_radius = np.zeros_like(skeleton)
    
    # Iterate over each connected component in the skeleton map
    for region_id in np.unique(skeleton):
        if region_id == 0:  # Skip background
            continue

        # Get the coordinates of the current region
        y, x = np.where(skeleton == region_id)
        
        # Find the nearest zero value in the EDT for each point in the region
        for i, j in zip(y, x):
            # Update the minimum radius for each point in the region
            min_radius[i, j] = distances[i, j]
            min_radius = min_radius.astype(float)
            
            
    # Identify regions that satisfy the neck condition
    neck = np.where(min_radius < neck_thr, min_radius, 0)
    
    # Count the number of unique necks
    number_of_necks = count_necks(neck.copy())
    
    return number_of_necks

def iterative_dfs(image, x, y):
    """
    Perform a depth-first search to find connected components (clusters) in an image.
    
    Parameters:
    - image: A 2D array representing the image.
    - start_x: Starting x-coordinate for the DFS.
    - start_y: Starting y-coordinate for the DFS.
    
    Returns:
    - None
    """
    # Define directions including diagonals
    directions = [
        (0, 1),  # East
        (1, 0),  # South
        (0, -1),  # West
        (-1, 0),  # North
        (-1, -1),  # North-West
        (1, 1),  # South-East
        (-1, 1),  # North-East
        (1, -1)   # South-West
    ]
    
    # Perform an iterative depth-first search starting from pixel (x, y)
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cy < 0 or cx >= len(image[0]) or cy >= len(image):
            continue
        if image[cy][cx] == 0:  # Assuming 0 represents non-clustered pixels
            continue
        if image[cy][cx] == -1:  # Already visited
            continue
        
        # Mark the current pixel as visited
        image[cy][cx] = -1  # Use -1 or another value to indicate visited pixels
        
        # Add all connected pixels to the stack
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < len(image[0]) and 0 <= ny < len(image) and image[ny][nx]!= 0:
                stack.append((nx, ny))

def count_necks(neck_map):
    """Count the number of necks in the ridge map."""
    num_necks = 0
    for y in range(len(neck_map)):
        for x in range(len(neck_map[0])):
            if neck_map[y][x]!= 0 and neck_map[y][x]!= -1:
                num_necks += 1
                iterative_dfs(neck_map, x, y)
    
    return num_necks

########################################################################################################################################################
                                              
########################################################################################################################################################
def count_segmented_layer(image, interaction, equidistant_files, t_step, rescaling_factor):
    
    global saving_folder
   
    retval, labels = cv2.connectedComponents(image)
    
    if interaction in equidistant_files:
        
        areas = np.zeros(retval, dtype=int)
        circularities = []
        
        
        
        # Calculate the area of each component
        for label in range(1, retval):  # start from 1 to skip the background
            areas[label] = np.sum(labels == label)
        areas = areas*rescaling_factor
            
            # Creating mask for finding the contour of the single object
        for label in range(1, retval):  # start from 1 to skip the background
            mask = np.uint8(labels == label)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            if contours: 
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = (4 * np.pi * areas[label]) / (perimeter ** 2)
                    circularities.append(circularity)
                    
        circularities = [round(num, 3) for num in circularities]
        regions = [i+1 for i in range(len(circularities))]            
        
        ''' NOTE: Altough a perfect circle has a circularity of 1, some object can have a value even greater. This is due to computing problem: the perimeter
        of the objects is computed with a summation of number of squared and quantized pixels, thus the formula 4*pi*A/P^2 MUST fail. 
        To obtain a better estimates, it is necessary to find a better estimate of the perimeter '''
        
        # Map component labels to hue value
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        
        # Convert to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        
        # Set background label to black
        labeled_img[label_hue == 0] = 0
        
        # Find the center of each component and add text
        for label in range(1, retval):
            mask = (labels == label)
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                x_center = int(np.mean(x))
                y_center = int(np.mean(y))
                text = f'{label}'
                cv2.putText(labeled_img, text, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
        
        if interaction == 0:
            filename = t_step
        else:
            filename = interaction * t_step
            
        path = os.path.join(saving_folder, f"t={filename} s.png")
        cv2.imwrite(path, labeled_img)
        
        df = pd.DataFrame(data = {'Regions': regions, 'Areas (nm^2)': areas[1:], 'Circularity': circularities})
        df["Circularity"] = df["Circularity"].astype(float).round(3).map("{:,.2f}".format)
        df['Areas (nm^2)'] = df['Areas (nm^2)'].astype(float).round(3).map("{:,.2f}".format)
        styled_df = df.style.hide(axis='index').set_table_styles([
        {'selector': 'tr:last-child td', 'props': [('border-bottom', '2px solid black'), ('padding-bottom', '10px')]},
        {'selector': 'th', 'props': [('border-bottom', '4px solid black'), ('padding-bottom', '10px')]},  # Adjusted border-bottom thickness
        {'selector': 'td', 'props': [('border-top', '2px solid black'), ('padding-top', '10px')]},
        {'selector': 'th', 'props': [('text-align', 'center')]}
        ])
        
        file_path = f"{saving_folder}\\Table_t={filename} s.png"
        dfi.export(styled_df, file_path, table_conversion = "chrome")
        
    return retval-1
    
    
    
    
    
    
    