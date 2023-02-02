# Function form of the above code
import pyFAI
import pyFAI.calibrant
import numpy as np
import matplotlib.pyplot as plt
from pyFAI.gui import jupyter
import scipy.ndimage as ndimage
import pyFAI.detectors
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

def plot_calibration_image(distance, wavelength, cmap):
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
    ai_short = AzimuthalIntegrator(dist=distance, poni1=poni1, poni2=poni2, detector=dete, wavelength=wavelength)

    # Generate the calibration image
    img = cal.fake_calibration_image(ai_short)

    # Plot the calibration image
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=cmap)
    plt.title(calibrant)
    plt.show()
    
    return img
    

def generate_2D_image(nspots, width, offset, size=4, shape = 'square', cmap = 'viridis'):
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

def generate_noisemap(combined_image, intensity, cmap = 'viridis'):
    """
    This function generates a noise map to overlay on top of the combined image, the noise map should be the same size as the combined image and are random values between 0 and 1 scalars that are multiplied by the combined image.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
        intensity (float): The intensity of the noise map.
    """
    #generate a noise map to overlay on top of the combined image, the noise map should be the same size as the combined image and are random values between 0 and 1 scalars that are multiplied by the combined image
    #the user can specify how much impact the noise map has on the combined image by specifying the intensity of the noise map
    noise_map = np.random.rand(2048,2048) * intensity
    
    #add the noise map to the combined image
    combined_image_with_noise = combined_image + noise_map
    
    #display the combined image with the noise map
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_image_with_noise, cmap=cmap)
    plt.title("Combined Image with Noise Map")
    plt.show()
    
    return combined_image_with_noise

def create_numpy_array_mask(combined_image, width):
    """
    This function creates a mask for the azimuthal integrator to mask everything but an area of interest.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
        width (int): The width of the line of interest.
    """
    #create a mask for the azimuthal integrator to mask everything but an area of interest.
    #this area of interest is a line of user specified width that starts at the center of the image and extends to the edge of the image
    #the mask starts at the center and only goes positive in the y direction
    mask = np.zeros_like(combined_image)
    mask[1024-width:1024+width, 1024:] = 1
    
    #display the mask
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='viridis')
    plt.title("Mask")
    plt.show()
    
    return mask

#Create a function that takes the combined image and integrates it using the azimuthal integrator and displays the 1D image
def integrate_image(combined_image, distance, wavelength, resolution = 3000, mask = None):
    """
    This function integrates the combined image using the azimuthal integrator and displays the 1D image.
    
    Parameters:
        combined_image (2D array): The image of the combined spots and calibration.
    """
    #initialize the azimuthal integrator
    
     # Initialize the detector
    dete = pyFAI.detectors.Perkin()
    p1, p2, p3 = dete.calc_cartesian_positions()
    poni1 = p1.mean()
    poni2 = p2.mean()
    
    
    ai = AzimuthalIntegrator(dist=distance, poni1=poni1, poni2=poni2, detector=dete, wavelength=wavelength)
    
    #integrate the combined image using the azimuthal integrator
    q, I = ai.integrate1d(combined_image, resolution, unit = 'q_A^-1', mask = mask)
    
    #plot the 1D image
    plt.figure(figsize=(10, 10))
    plt.plot(q, I)
    plt.title("1D X-Ray Diffraction Image")
    plt.show()
    
    return q, I

def mask_rotation(mask, angle):
    """
    This function rotates the create mask by a user specified angle amount, if the angle specified is 1, the result is that the mask is rotated by one degree.
    
    Parameters:
        mask (2D array): The mask to use for the integration.
        angle_of_rotation (int): The angle of rotation.
        """
    rotated_mask = ndimage.rotate(mask, angle, reshape = False, mode = 'mirror')
        
        
    #display the rotated mask
    plt.figure(figsize=(10, 10))
    plt.imshow(rotated_mask, cmap='viridis')
    plt.title("Rotated Mask")
    plt.show()
    
    return rotated_mask
