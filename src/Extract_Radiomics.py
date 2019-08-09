import SimpleITK as sitk
import numpy as np
import pandas as pd
import multiprocessing
import os, sys
from tqdm import tqdm
from radiomics import featureextractor

def worker():
    print('Worker')
    return

NumCPU=10

if __name__ == '__main__':
    jobs = []
    for i in range(NumCPU):
        p = multiprocessing.Process(target=worker)
        jobs.append(p)
        p.start()

# ##################################################
# Setting
ImageFolderPath='/data/Brain/disorder/dataset/400_coT1_nii/Healthy_coT1/'
MaskFolderPath='/data/Brain/disorder/dataset/400_coT1_nii/mask/Healthy_coT1/'

RadiomicsFile='./../results/Radiomics_Healthy_coT1.xlsx'
ErrorFile='./../error/Extract_Features.xlsx'

Files=[file for file in os.listdir(ImageFolderPath) if file.endswith('.nii')]

Features={}
Features['FeaturesName']=[]
NameInitial=True
Error=[]

# ##################################################
# Extract Features
extractor=featureextractor.RadiomicsFeaturesExtractor()
for file in tqdm(Files):
    MaskPath=MaskFolderPath+file[:-4]+'.npy'
    ImagePath=ImageFolderPath+file

    try:
        print('Load Mask and Image from ',file[:-4])
        Mask=np.load(MaskPath)
        sitkImage=sitk.ReadImage(ImagePath)
        npImage=sitk.GetArrayFromImage(sitkImage)
        npImage=npImage.transpose(2,1,0)

        print('Extract the features...')
        result=extractor.execute(sitk.Cast(sitk.GetImageFromArray(npImage),sitk.sitkUInt64),
                                 sitk.GetImageFromArray(Mask.astype(int)))
        print("Done")

        Features[file]=[]
        for Key, Value in result.items():
            if NameInitial:
                Features['FeaturesName'].append(Key)
                Features[file].append(Value)
            else:
                Features[file].append(Value)
        NameInitial=False

    except:
        print('Fail to extract the features from ',file)
        Error.append(file)

# ##################################################
# Save Features
p=pd.DataFrame(Features)
p.set_index('FeaturesName')
p=p.T
p.to_excel(RadiomicsFile)
print('Save features to ',RadiomicsFile)
# Save Error
p=pd.DataFrame(Error)
p.to_excel(ErrorFile)
print('Save error message to ',ErrorFile)

