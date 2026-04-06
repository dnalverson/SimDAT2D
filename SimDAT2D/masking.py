import SimDAT2D as sim
import numpy as np
import numpy.ma as ma
import numpy.random as rnd
import matplotlib.pyplot as plt
import pyFAI
import pyFAI.azimuthalIntegrator as AI
import pandas as pd
import dask
from dask import delayed
from scipy.ndimage import rotate as scipy_rotate

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