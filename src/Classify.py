import os, sys
sys.path.append('./../')
import numpy as np
import SimpleITK as sitk
import pandas as pd
import tensorflow as tf
from include.Model import *
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

np.random.seed(18)
tf.logging.set_verbosity(tf.logging.DEBUG) # DEBUG，INFO，WARN，ERROR, FATAL

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# ###############################################
# FolderPath
H_Path='/data/Brain/disorder/dataset/400_coT1_nii/Healthy_coT1/'
S_Path='/data/Brain/disorder/dataset/400_coT1_nii/Schizo_coT1/'
# Set Files Extension that we want to collect
FileExtension='.nii'
# Collect All Images Names
H_Files=[file for file in os.listdir(H_Path) if file.endswith(FileExtension)]
S_Files=[file for file in os.listdir(S_Path) if file.endswith(FileExtension)]


Train=True
ResultsFile='./../results/ClassifyResults.xlsx'

rate_train=0.7
rate_validation=0.1
rate_test=0.2

n_epochs=100
batch_size=10
learning_rate=0.00001
save_model_path='./../models/classify_model'

Test=False
load_model_path=save_model_path+'.meta'
load_par_path='./../models/r_00001/'
threshold=0.5
save_test_path='./../results/testing_results_r_00001.xlsx'

# ###############################################
# Collect All Data
# Data=[sitk.GetArrayFromImage(sitk.ReadImage(Path+file)) for file in Files[:5]]
initial=True
for count in range(2):
    if count==0:
        Files=H_Files
        Path=H_Path
    elif count==1:
        Files=S_Files
        Path=S_Path
    for file in tqdm(Files):
        src=Path+file
        sitkImage=sitk.ReadImage(src)
        if initial:
            Data=[sitk.GetArrayFromImage(sitkImage)]
            size=list(sitkImage.GetSize())
            if count==0:
                Label=[0]
            elif count==1:
                Label=[1]
            initial=False
        else:
            Data.append(sitk.GetArrayFromImage(sitkImage))
            temp_size=list(sitkImage.GetSize())
            if count==0:
                Label.append(0)
            elif count==1:
                Label.append(1)
            for i in range(len(size)):
                if temp_size[i] < size[i]:
                    size[i]=temp_size[i]
        del sitkImage

# ###############################################
# Shuffule Data
num_case=len(Label)
Numbers=np.arange(num_case)
np.random.shuffle(Numbers)
temp_Data=Data.copy()
temp_Label=Label.copy()
del Data, Label

initial=True
size=size[::-1]
for index in tqdm(range(len(Numbers))):
    i=Numbers[index]
    start=((np.array(temp_Data[i].shape)-np.array(size))/2).astype(int)
    end=(start+np.array(size))
    if initial:
        Data=[temp_Data[i][start[0]:end[0],start[1]:end[1],start[2]:end[2]]]
        Label=[temp_Label[i]]
        initial=False
    else:
        Data.append(temp_Data[i][start[0]:end[0],start[1]:end[1],start[2]:end[2]])
        Label.append(temp_Label[i])
del temp_Data, temp_Label

# ###############################################
# Cut Training, Validation and Testing Sets
Train_X=np.array(Data[:int(num_case*rate_train)])
Train_Y=np.array(Label[:int(num_case*rate_train)])
Train_X=np.expand_dims(Train_X,axis=-1)
Train_Y=np.expand_dims(Train_Y,axis=-1)

Validation_X=np.array(Data[int(num_case*rate_train):int(num_case*(rate_train+rate_validation))])
Validation_Y=np.array(Label[int(num_case*rate_train):int(num_case*(rate_train+rate_validation))])
Validation_X=np.expand_dims(Validation_X,axis=-1)
Validation_Y=np.expand_dims(Validation_Y,axis=-1)

Test_X=np.array(Data[int(num_case*(rate_train+rate_validation)):])
Test_Y=np.array(Label[int(num_case*(rate_train+rate_validation)):])
Test_X=np.expand_dims(Test_X,axis=-1)
Test_Y=np.expand_dims(Test_Y,axis=-1)

# ###############################################
# Load Model and Training
if Train:
    Record=cnn_model(images=Train_X,
                     labels=Train_Y,
                     val_images=Validation_X,
                     val_labels=Validation_Y,
                     n_epochs=n_epochs,
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     save_model_path=save_model_path)
    print("Save model to ",save_model_path)
    p=pd.DataFrame(Record)
    p.to_excel(excel_writer=ResultsFile,index=False)
    print("Save result file to ",ResultsFile)

# ###############################################
# Testing
if Test:
    ans=predict(images=Test_X,
                model_path=load_model_path,
                par_path=load_par_path)

    ans[ans>=threshold]=1
    ans[ans<threshold]=0

    matrix=confusion_matrix(Test_Y,ans)
    acc=(matrix[0,0]+matrix[1,1])/len(Test_Y)

    print("Test Accuracy: ", acc)
    print("Confusion Matrix: ", matrix)

    df=pd.DataFrame(matrix)
    df.to_excel(save_test_path,index=False)
    print("Save results to ", save_test_path)

