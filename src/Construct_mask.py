import os, sys
import nipy.labs.mask
import scipy.ndimage.morphology
import numpy as np
from tqdm import tqdm

# ########################################
# Setting
FolderPath='/data/Brain/disorder/dataset/400_coT1_nii/Schizo_coT1/'
SavedPath='/data/Brain/disorder/dataset/400_coT1_nii/mask/Schizo_coT1/'

Files=[file for file in os.listdir(FolderPath) if file.endswith('.nii')]

for file in tqdm(Files):
    src=FolderPath+file
    dst=SavedPath+file[:-4]
    print('Generate brain mask from ',file)
    brain_mask=nipy.labs.mask.compute_mask_files(src)
    print('Full the holes in mask...')
    full_mask=scipy.ndimage.morphology.binary_fill_holes(brain_mask)
    np.save(dst,full_mask)
    print('Save brain mask to ',dst+'.npy')


