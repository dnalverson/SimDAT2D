import IsoDAT2D as iso
import SimDAT2D as sim
import dask.array as da
import dask.dataframe as dd
import masking
import matplotlib.pyplot as plt
import numpy as np
from dask import compute, delayed

data = np.load(my_sim_data.npy)

print('Data loaded')

array, ai_pe = masking.make_chi_array(data, .4, .4e-10)

print('Chi array created')

def make_masks(array, slices):
    masks = []
    for i in slices:
        masks.append(masking.generate_mask_slices(array, 5, i, offset = 10))
        print('Mask with {} slices created'.format(i))
    return masks

slices = da.arange(1, 5, 1)
my_masks = make_masks(array, slices)

print('Masks created')

masks = []
for i in range(len(my_masks)):
  masks.append(masking.rotate_mask_360(my_masks[i], 360, 25, plot = False))
  print('Mask {} rotated'.format(i))
  
print('Masks rotated')

flat_list = [element for inner_list in masks for element in inner_list]

my_masks = np.save('my_masks.npy', flat_list)
print('Masks saved')

def integrate_image(combined_image, distance, wavelength, resolution = 3000, mask = None, show = False, radial_range = None):
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
    q, I = ai.integrate1d(combined_image, resolution, radial_range = radial_range, unit = 'q_A^-1', mask = mask)
    
    if show == True:
        #plot the 1D image
        plt.figure(figsize=(10, 10))
        plt.plot(q, I)
        plt.title("1D X-Ray Diffraction Image")
        plt.show()

df = pd.DataFrame()
for i in range(len(flat_list)):
    q, I = integrate_image(data, .4, .4e-10, 1000, flat_list[i], radial_range = (1, 7))
    df[i] = I
    print('Integration {} of {} complete'.format(i+1, len(flat_list)))
    
df.to_csv('integrations.csv', index = False)
np.save('q.npy', q)
print('Integrations saved')

plt.figure(figsize=(10,10))
#plot the integrations in a waterfall plot where each integration is a row
plt.imshow(integrations, cmap = 'magma', aspect = 'auto')
plt.savefig('waterfall.png')
    

