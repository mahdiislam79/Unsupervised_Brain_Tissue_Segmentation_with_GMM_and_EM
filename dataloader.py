import os
import nibabel as nib
import numpy as np

def load_mri_data(input_dir):
    t1_file = os.path.join(input_dir, 'T1.nii')
    flair_file = os.path.join(input_dir, 'T2_FLAIR.nii')
    gt_file = os.path.join(input_dir, 'LabelsForTesting.nii')

    t1_data = nib.load(t1_file).get_fdata()
    flair_data = nib.load(flair_file).get_fdata()
    gt_data = nib.load(gt_file).get_fdata()

    return t1_data, flair_data, gt_data

def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)
