import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from rpy2.robjects import DataFrame, FloatVector, IntVector
from rpy2.robjects.packages import importr
from math import isclose

# #################################################
# Setting
H_FeaturesFile='./../results/Radiomics_Healthy_WM.xlsx'
S_FeaturesFile='./../results/Radiomics_Schizo_WM.xlsx'

SaveICC='./../results/ICC_WM.xlsx'
SaveUsefulFeature='./../results/SelectedFeatures.xlsx'

StartFeatures=22    # Start from meaingful feature

# #################################################
# Load Features
H_df=pd.read_excel(H_FeaturesFile)
S_df=pd.read_excel(S_FeaturesFile)

np_H_df=np.array(H_df)
np_S_df=np.array(S_df)


# Mark Groups
Features=[]
Groups=[]
print("Marking groups...")
for i in range(2):
    if i==0:
        for F_num in np.arange(StartFeatures,np_H_df.shape[1]):
            Features.append(list(np_H_df[1:,F_num]))
            Groups.append(list(np.zeros(np_H_df.shape[0]-1).astype(int)))
    if i==1:
        for S_num in np.arange(StartFeatures,np_S_df.shape[1]):
            count=S_num-StartFeatures
            Features[count]=Features[count]+list(np_S_df[1:,S_num])
            Groups[count]=Groups[count]+list(np.ones(np_S_df.shape[0]-1).astype(int))

# #################################################
# Extract ICC
icc_features={}
icc_features['FeatureName']=np_H_df[0,StartFeatures:]
icc_features['ICC_value']=[]
r_icc=importr("ICC")
print('Computing ICC values...')
for i in tqdm(range(len(Features))):
    df=DataFrame({"groups":IntVector(Groups[i]),
                  "values":FloatVector(Features[i])})
    icc_res=r_icc.ICCbare("groups", "values", data=df)
    icc_features['ICC_value'].append(icc_res[0])
print("Done")

# Save ICC Values
p=pd.DataFrame(icc_features)
p.to_excel(SaveICC)
print('Save ICC values to ',SaveICC)

