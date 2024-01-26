import IsoDAT2D as iso
import SimDAT2D as sim
import dask.array as da
import dask.dataframe as dd
import masking
import matplotlib.pyplot as plt
import numpy as np
from dask import compute, delayed

data = np.load(simulated_data.npy)

print('Data loaded')

array, ai_pe = masking.make_chi_array(data, .4, .4e-10)

print('Chi array created')

def make_masks(array, slices):
    masks = []
    for i in slices:
        masks.append(masking.generate_mask_slices(array, 1, i, 8))
        print('Mask with {} slices created'.format(i))
    return masks

slices = da.arange(1, 25, 1)
my_masks = make_masks(array, slices)

print('Masks created')

masks = []
for i in range(len(my_masks)):
  masks.append(masking.rotate_mask_360(my_masks[i], 360, 1, plot = False))
  print('Mask {} rotated'.format(i))
  
print('Masks rotated')

flat_list = [element for inner_list in masks for element in inner_list]

integrations = []
for i in range(len(flat_list)):
    integrations.append(sim.integrate_image(data, .4, .4e-10, 1000, flat_list[i]))
    print('Integration {} of {} complete'.format(i+1, len(flat_list)))
    
    plt.figure(figsize=(10,10))
    #plot the integrations in a waterfall plot where each integration is a row
    plt.imshow(integrations, cmap = 'magma', aspect = 'auto')
    plt.savefig('waterfall.png')
    
