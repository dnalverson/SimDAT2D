# Copyright (c) 2023, Danielle N. Alverson
# All rights reserved.
#
# This software is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for details.


import pyFAI
import pyFAI.calibrant
import numpy as np
import matplotlib.pyplot as plt
from pyFAI.gui import jupyter
import scipy.ndimage as ndimage
import pyFAI.detectors
from PIL import Image
import numpy as np
import numpy.ma as ma
import numpy.random as rnd
import pyFAI.azimuthalIntegrator as AI
import pandas as pd
import dask
from dask import delayed
from scipy.ndimage import rotate as scipy_rotate


def create_iso_no_input(distance, wavelength, cmap, calib = 22):
    ''' This function creates a calibration image for a given calibrant, distance, and wavelength. 
    The user does not have to input the calibrant, it is already specified in the function. '''
    
    calibrants = [ "AgBh", "Al", "alpha_Al2O3", "Au", "C14H30O", "CeO2", "Cr2O3",
                  "cristobaltite", "CrOx", "CuO", "hydrocerussite", "LaB6", "LaB6_SRM660a",
                  "LaB6_SRM660b", "LaB6_SRM660c", "mock", "NaCl", "Ni", "PBBA",
                  "Pt", "quartz", "Si", "Si_SRM640", "Si_SRM640a", "Si_SRM640b",
                  "Si_SRM640c", "Si_SRM640d", "Si_SRM640e", "TiO2", "ZnO" 
                  ]
    calibrant = calibrants[calib]
    cal = pyFAI.calibrant.ALL_CALIBRANTS(calibrant)
    
      # Initialize the detector
    dete = pyFAI.detectors.Perkin()
    p1, p2, p3 = dete.calc_cartesian_positions()
    poni1 = p1.mean()
    poni2 = p2.mean()

    # Initialize the azimuthal integrator
    ai_short = AI.AzimuthalIntegrator(dist=distance, poni1=poni1, poni2=poni2, detector=dete, wavelength=wavelength)

    # Generate the calibration image
    img = cal.fake_calibration_image(ai_short)

    # Plot the calibration image
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=cmap)
    plt.title(calibrant)
    plt.show()
    
    return img



def create_isotropic(distance, wavelength, cmap):
    """
    This function plots a calibration image for a given calibrant, distance, and wavelength.

    Parameters:
        calibrant (str): The name of the calibrant to use (e.g. 'Si', 'YAG').
        distance (float): The distance from the sample to the detector, in meters.
        wavelength (float): The wavelength of the x-rays, in 1e-10 meters.
    """
    calibrants = [
    "AgBh", "Al", "alpha_Al2O3", "Au", "C14H30O", "CeO2", "Cr2O3", 
    "cristobaltite", "CrOx", "CuO", "hydrocerussite", "LaB6", "LaB6_SRM660a", 
    "LaB6_SRM660b", "LaB6_SRM660c", "mock", "NaCl", "Ni", "PBBA", 
    "Pt", "quartz", "Si", "Si_SRM640", "Si_SRM640a", "Si_SRM640b", 
    "Si_SRM640c", "Si_SRM640d", "Si_SRM640e", "TiO2", "ZnO"
    ]
    
     # Retrieve the specified calibrant
    print("Select calibrant:")
    for i, calibrant in enumerate(calibrants):
        print(f"{i+1}. {calibrant}")
    selected_calibrant = int(input("Enter the number of the selected calibrant: ")) - 1
    calibrant = calibrants[selected_calibrant]
    print(calibrant)
    cal = pyFAI.calibrant.ALL_CALIBRANTS(calibrant)
    

    # Initialize the detector
    dete = pyFAI.detectors.Perkin()
    p1, p2, p3 = dete.calc_cartesian_positions()
    poni1 = p1.mean()
    poni2 = p2.mean()

    # Initialize the azimuthal integrator
    ai_short = AI.AzimuthalIntegrator(dist=distance, poni1=poni1, poni2=poni2, detector=dete, wavelength=wavelength)

    # Generate the calibration image
    img = cal.fake_calibration_image(ai_short)

    # Plot the calibration image
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=cmap)
    plt.title(calibrant)
    plt.show()
    
    return img
    

def create_anisotropic(nspots, width, offset, size=4, shape = 'square', cmap = 'viridis'):
    """
    This function generates a 2D image of small spots on a detector image that's 2048x2048 pixels. Each spot is separated by a width and offset and has the highest intensity in the center of the spot and the 
    lowest intensity at the edges of the spot. The spots are 4x4 pixels in size. The function should take the number of spots, the width of the spots, and the offset between spots as parameters.

    Parameters:
        nspots (int): The number of spots to generate.
        width (float): The width of the spots.
        offset (float): The offset between spots.
    """
    # initialize the detector image
    detector_image = np.zeros((2048, 2048))
    
    
    #generate the spot where the highest intensity is in the center of the spot and the lowest intensity is at the edges of the spot, the intensity values are specified by the user as a parameter
    #the spots at a higher intensity at the center of the image and a lower intensity at the edges of the image but still have a gaussian distribution within the spot. 
    #the spots can either be square or cricular in shape and this is specified by the user, the default is square. This will start as a for loop.
    for i in range(nspots):
        for j in range(nspots):
            if shape == 'square':
                x = i * offset
                y = j * offset
                for k in range(size):
                    for l in range(size):
                        if x + k < 2048 and y + l < 2048:
                            detector_image[x+k][y+l] = np.exp(-((k-size/2)**2 + (l-size/2)**2) / (2 * width**2))
                    
    # display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(detector_image, cmap=cmap)
    plt.title("2D X-Ray Diffraction Image")
    plt.show()
    
    return detector_image


def rotate_image(image, angle, cmap = 'viridis'):
    """
    This function rotates the image by a specified angle.
    
    Parameters:
        image (2D array): The image to be rotated.
        angle (float): The angle to rotate the image by.
    """
    #rotate the image by a specified angle
    rotated_image = ndimage.rotate(image, angle, reshape = False, mode = 'wrap')
    
    #expanding the image so the spots can extend beyond the original image contraints of 2048x2048 pixel
    
    
    #display the rotated image
    plt.figure(figsize=(10, 10))
    plt.imshow(rotated_image, cmap=cmap)
    plt.title("Rotated Image")
    plt.show()
    
    return rotated_image
    
#create a function that combines the image created from the generate 2D image function and the calibration image created from the plot calibration image function and creates one 2D image 
#with the spots and calibratation image

def combine_image(spot_image, calibration_image, cmap='viridis'):
    """
    This function combines the spot image and the calibration image into one 2D image.
    
    Parameters:
        spot_image (2D array): The image of the spots.
        calibration_image (2D array): The image of the calibration.
    """
    #combine the spot image and the calibration image into one 2D image
    combined_image = spot_image + calibration_image
    
    #display the combined image
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_image, cmap=cmap)
    plt.title("Combined Image")
    plt.show()
    
    return combined_image

#Generate a noise map to overlay on top of the combined image, the noise map should be the same size as the combined image and are random values between 0 and 1 scalars that are multiplied by the combined image
#the user can specify how much impact the noise map has on the combined image by specifying the intensity of the noise map

def generate_noisemap(combined_image, intensity = 'med', cmap = 'viridis'):
    """
    This function generates a noise map to overlay on top of the combined image, the noise map should be the same size as the combined image and are random values between 0 and 1 scalars that are multiplied by the combined image.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
        intensity (float): The intensity of the noise map.
    """
    
    if intensity == 'low':
        intensity = 2
    elif intensity == 'med':
        intensity = 5
    elif intensity == 'high':
        intensity = 7.5

    #generate a noise map to overlay on top of the combined image, the noise map should be the same size as the combined image and are random values between 0 and 1 scalars that are multiplied by the combined image
    #the user can specify how much impact the noise map has on the combined image by specifying the intensity of the noise map
    # Generate a normalized noise map
    noise_map = np.random.normal(loc=2, scale=intensity, size=(2048, 2048))

    # Normalize the noise map
    #normalized_noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))

    # Multiply the normalized noise map with your image
    result = noise_map * combined_image
    
    #display the combined image with the noise map
    plt.figure(figsize=(10, 10))
    plt.imshow(result, cmap=cmap)
    plt.title("Combined Image with Noise Map")
    plt.show()
    
    return result

def create_mask(combined_image, width):
    """
    This function creates a mask for the azimuthal integrator to mask everything but an area of interest.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
        width (int): The width of the line of interest.
    """
    
    #create a mask for the azimuthal integrator to mask everything but an area of interest.
    #this area of interest is a line of user specified width that starts at the center of the image and extends to the edge of the image
    #the mask starts at the center and only goes positive in the y direction
    mask = np.ones_like(combined_image)
    mask[1024-width:1024+width, 1024:] = 0
    
    #display the mask
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='viridis')
    plt.title("Mask")
    plt.show()
    
    return mask

def read_poni_file(poni_file):
    """
    This function reads a .poni file and extracts the distance and wavelength, rotation, and poni from the file.
    
    Parameters:
        poni_file (str): The path to the .poni file.
    """
    #read the .poni file and extract the distance and wavelength from the file
    with open(poni_file, 'r') as file:
        for line in file:
            if 'Distance:' in line:
                distance = float(line.split()[1])
            if 'Wavelength:' in line:
                wavelength = float(line.split()[1])
            if 'Rot1:' in line:
                rot1 = float(line.split()[1])
            if 'Rot2:' in line:
                rot2 = float(line.split()[1])
            if 'Rot3:' in line:
                rot3 = float(line.split()[1])
            if 'Poni1:' in line:
                poni1 = float(line.split()[1])
            if 'Poni2:' in line:
                poni2 = float(line.split()[1])
    
    return distance, wavelength, rot1, rot2, rot3, poni1, poni2

#Create a function that takes the combined image and integrates it using the azimuthal integrator and displays the 1D image
def integrate_image(combined_image, distance, wavelength, resolution = 3000, mask = None, show = False, radial_range = None, poni = None):
    """
    This function integrates the combined image using the azimuthal integrator and displays the 1D image.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
    """
    #initialize the azimuthal integrator
    dete = pyFAI.detectors.Perkin()
    
    if poni == None:
    
        # Initialize the detector
        p1, p2, p3 = dete.calc_cartesian_positions()
        poni1 = p1.mean()
        poni2 = p2.mean()
    
        ai = AI.AzimuthalIntegrator(dist=distance, poni1=poni1, poni2=poni2, detector=dete, wavelength=wavelength)
        
    else:
        distance, wavelength, rot1, rot2, rot3, poni1, poni2 = read_poni_file(poni)
        ai = AI.AzimuthalIntegrator(dist=distance, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=dete, wavelength=wavelength)
    
    #integrate the combined image using the azimuthal integrator
    q, I = ai.integrate1d(combined_image, resolution, radial_range = radial_range, unit = 'q_A^-1', mask = mask)
    
    if show == True:
        #plot the 1D image
        plt.figure(figsize=(10, 10))
        plt.plot(q, I)
        plt.title("1D X-Ray Diffraction Image")
        plt.show()
    
    return q, I

def mask_rotation(mask, angle, show = False):
    """
    This function rotates the create mask by a user specified angle amount, if the angle specified is 1, the result is that the mask is rotated by one degree.
    
    Parameters:
        mask (2D array): The mask to use for the integration.
        angle_of_rotation (int): The angle of rotation.
        """
    rotated_mask = ndimage.rotate(mask, angle, reshape = False, mode = 'mirror')
        
    
    if show == True:
        #display the rotated mask
        plt.figure(figsize=(10, 10))
        plt.imshow(rotated_mask, cmap='viridis')
        plt.title("Rotated Mask")
        plt.show()
    
    return rotated_mask

def image_rotation(image, angle, show = False):
    """
    This function rotates the combined image by a user specified angle amount, if the angle specified is 1, the result is that the combined image is rotated by one degree.
    
    Parameters:
        image (2D array): The image of the combined spots and calibration.
        angle_of_rotation (int): The angle of rotation.
        """
    pil_format = Image.fromarray(image)
    rotated_image = pil_format.rotate(angle)
    rotated_image = np.array(rotated_image)
    
    if show == True:
        #display the rotated image
        plt.figure(figsize=(10, 10))
        plt.imshow(rotated_image, cmap='viridis')
        plt.title("Rotated Image")
        plt.show()
    return rotated_image

def rotate_and_integrate(combined_image, angle_of_rotation, distance, wavelength, resolution = 3000, mask = None):
    """
    This function takes the combined image, the mask, the distance, the wavelength, and the resolution of integration, and rotates the combined image by a user specified angle amount, if the angle specified is 1, the result will be 360 integrations of the combined image, each integration will be rotated by 1 degree.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
        angle_of_rotation (int): The angle of rotation.
        distance (float): The distance from the detector to the sample.
        wavelength (float): The wavelength of the x-rays.
        resolution (int): The resolution of the integration.
        mask (2D array): The mask to use for the integration.
    """
    
    import pandas as pd 
    
    #create a dataframe to store the 1D integrations
    df = pd.DataFrame()
    
    #create a loop that rotates the combined image by the user specified angle amount and integrates the image
    for i in range(0, 360, angle_of_rotation):
        #rotate the mask for the combined image
        rotated_image = image_rotation(combined_image, i);
    
        
        #integrate the rotated image
        q, I = integrate_image(rotated_image, distance, wavelength, resolution, mask, show = False);
        
        #add the 1D integration to the dataframe
        df[i] = I
        
        #create a waterfall plot of the 1D integrations, where each dataset is moved up on the y axis by a multiple of .5
    plt.figure(figsize=(10, 10))
    for j in range(0, 360, angle_of_rotation):
            plt.plot(q, (df[j]+ j*.01), alpha = .55, c = 'black')
    plt.xlabel('q A $^(-1)$')
    plt.ylabel('Intensity')
    plt.title("Waterfall Plot of Rotated 1D X-Ray Diffraction Images")
    plt.show()        
    return q, df

    # Final Subtraction Program
# Date Created: 11/2/2023
# Author:Celia Mercier, Danielle Alverson

import csv
import pandas as pd
import matplotlib.pyplot as plt

def parse_file(file_path):
    """This serves as a helper function to the main subtraction function by "cleaning" up the data
    and removing the explanations at the start to get the numbers alone"""
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Removes lines starting with '#'
            if not line.startswith('#'):
                values = line.strip().split()
                # Converts string in .txt to float
                x = float(values[0])
                y = float(values[1])
                

                data[x] = y
    return data


def subtract_and_store(total_signalfile_path, substrate_filepath):
    """This takes in text files containing coordinates of the integrations and subtracts the substrate from 
    the total signal. While subtracting it is keeping track of the smallest distance in between two points and
     and when it finishes parsing this process, it goes through the dictionary containg the subtracted values and 
      subracts the minumum value found, bring it as close to zero as possible """
    # Gets data from inputs
    data1 = parse_file(total_signalfile_path)
    data2 = parse_file(substrate_filepath)
    # Arbitrarily large number
    min_val = 1000

    result_dict = {}
    # Goes through x,y coordinates for total signal and finds matching x in substrate
    for x, y1 in data1.items():
        if x in data2:
            y2 = data2[x]
            #Subtracts y value of subtrate from total signal
            result_value = y1 - y2
            result_dict[x] = result_value

            if result_value < min_val and result_value != 0:
                min_val = result_value
                
    return result_dict

def save_data(path, result_dict):
    """A function to save the data as a csv by taking in the output of subtract_and_store"""
    final_sub_path = path
    # Opens csv file to place values in
    with open(final_sub_path, mode = 'w', newline = '' ) as file:
        # Space as delimiter
        writer = csv.writer(file, delimiter = ' ')
        for x in result_dict:
            # Subtracts by smallest space
            if result_dict[x] >= min_val:
                result_dict[x] = result_dict[x] - min_val
                writer.writerow([x, result_dict[x]])
                
def make_substrate_dict(substrate_filepath):
    """A function to make a dictionary of the substrate by taking in the filepath of the substrate"""
    substrate_dict = parse_file(substrate_filepath)
    return substrate_dict
                
def plot_data(result_dict, substrate_dict):
    """A function to plot the data by taking in the output of subtract_and_store"""
    # Separate coordinates into lists
    keys = list(result_dict.keys())
    values = list(result_dict.values())
    keys_data1 = list(substrate_dict.keys())
    values_data1 = list(substrate_dict.values())
    # Plotting data from the first file
    plt.figure()
    # Plotting data from the first file
    plt.plot(keys_data1, values_data1, label='Total Signal', color='blue')
    # Plotting data from the second file
    plt.plot(keys, values, label='Subtraction', color='magenta')
    plt.xlabel("q(A^-1)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()      

def make_chi_array(tiff, dist, wavelength):
    ''' Returns a chi array from a tiff file.
    
    Keyword arguments:
    tiff -- tiff file
    dist -- distance from the sample to the detector
    wavelength -- wavelength of the x-rays
    '''
    
    #creating the detector object and calculating the cartesian positions
    dete = pyFAI.detectors.Perkin()
    p1, p2, p3 = dete.calc_cartesian_positions()
    poni1 = p1.mean()
    poni2 = p2.mean()
    
    #creating the azimuthal integrator object
    ai_pe = AI.AzimuthalIntegrator(dist=dist, wavelength=wavelength, poni1=poni1, poni2=poni2, detector=dete)
    
    #creating the chi array
    chi_array = np.degrees(ai_pe.chiArray())
    
    return chi_array, ai_pe

def slice_mask(chi, width, plot = False):
    ''' Returns a mask that has one slice of the chi array left unmasked to be used for integration.

    Keyword arguments:
    chi -- chi array 
    width -- width of the slice in degrees
    plot -- if True, plots the mask (default False)
    '''
    #creating the mask and setting the first width pixels to False to leave a slice of width pixels unmasked
    mask = np.ma.masked_outside(chi, 0, width)
    
    if plot == True:
        plt.figure()
        plt.imshow(mask, cmap='magma')
    
    return mask

def rotate_mask_360(chi_array, width, rot_degree = 1, plot = False):
    ''' Rotates a mask 360 degrees around the center of the mask.
    
    Keyword arguments:
    mask -- mask to be rotated
    chi_array -- chi array to be used for the rotation
    '''
    #creating a list of angles to rotate the mask
    
    positive_angles = np.arange(0, 179, rot_degree)
    negative_angles = np.arange(-179, 0, rot_degree)
    
    #creating a list of rotated masks
    rotated_masks = []
    
    #rotating the mask and appending it to the list of rotated masks
    for angle in positive_angles:
        rotated_masks.append(np.ma.masked_outside(chi_array,0+angle, width+angle))
        
        if plot == True:
            plt.figure()
            plt.imshow(rotated_masks[-1], cmap = 'magma')
        
    for angle in negative_angles:
        rotated_masks.append(np.ma.masked_outside(chi_array,0+angle, width+angle))
        
        if plot == True:
            plt.figure()
            plt.imshow(rotated_masks[-1], cmap = 'magma')
    
    return rotated_masks

def generate_mask_slices(array, width, num_slices, offset = 5):
    
    ''' Returns a mask with multiple slices of the chi array left unmasked to be used for integration.
    
    Keyword arguments:
    chi_array -- chi array
    width -- width of the slice in degrees
    num_slices -- number of slices
    offset -- offset between slices in degrees
    plot -- if True, plots the mask (default False)
    
    '''
    mask_list = []
    
    # Create masks for the positive values
    for i in range(num_slices):
        start = i * (width + offset)
        end = start + width
        mask_list.append(ma.masked_inside(array, start, end))

    # Create masks for the negative values
    for i in range(num_slices):
        start = - (i + 1) * (width + offset)
        end = start - width
        mask_list.append(ma.masked_inside(array, start, end))
    
    #add all genrated masks together
    
    print(mask_list)

    combined_mask = mask_list[0]
    for mask in mask_list[1:]:
        combined_mask += mask
        
    inverted_mask = ~combined_mask.mask 
    plt.figure()
    plt.imshow(~combined_mask.mask)
    
    return inverted_mask

def rotate_generated_mask(mask, deg, offset = 1, plot = False):
    rotated_masks = []
    deg_list = np.arange(0, deg, offset)  # Adjust the range as needed
    for i in deg_list:
        rotated_mask = ma.masked_invalid(scipy_rotate(mask, angle=i, reshape=False, order=0))
        rotated_masks.append(rotated_mask)
        if plot == True:  
         plt.figure()
        
         plt.imshow(rotated_mask)
    return rotated_masks          
