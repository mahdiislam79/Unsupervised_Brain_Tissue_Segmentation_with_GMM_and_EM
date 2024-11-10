import os
import numpy as np
import pandas as pd
import nibabel as nib
from em_algorithm import em_algorithm
from dataloader import load_mri_data, normalize_data
from segmentation import generate_segmentation_mask, save_segmentation
from evaluation import dice_score
import matplotlib.pyplot as plt

def process_em_for_folder(input_dir, output_dir):
    # Load and normalize MRI data
    t1_data, flair_data, gt_data = load_mri_data(input_dir)
    t1_data, flair_data = normalize_data(t1_data), normalize_data(flair_data)
    
    # Extract data for clustering
    y_T1, y_T2 = t1_data[gt_data > 0], flair_data[gt_data > 0]
    data = np.array([y_T1, y_T2]).T

    # Run EM algorithm with k-means initialization handled internally
    mu, sigma, pi, log_likelihoods, responsibilities = em_algorithm(data)

    # Generate segmentation mask
    mapped_labels = generate_segmentation_mask(data, responsibilities, mu)
    segmentation = np.zeros_like(gt_data)
    segmentation[gt_data > 0] = mapped_labels
    save_segmentation(output_dir, segmentation, nib.load(os.path.join(input_dir, 'T1.nii')).affine)

    # Calculate Dice scores
    dice_scores = {label: dice_score(segmentation, gt_data, label) for label in range(1, 4)}

    # Plot and save log-likelihood evolution
    plt.figure()
    plt.plot(log_likelihoods, color='blue')
    plt.title("Log-Likelihood Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.savefig(os.path.join(output_dir, "log_likelihood.png"))
    plt.close()

    return {'mu': mu, 'sigma': sigma, 'dice_scores': dice_scores}

def process_all_folders_with_em(parent_directory, output_parent_directory):
    all_em_results = {}
    for folder_name in os.listdir(parent_directory):
        input_dir = os.path.join(parent_directory, folder_name)
        if os.path.isdir(input_dir):
            output_dir = os.path.join(output_parent_directory, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            try:
                em_results = process_em_for_folder(input_dir, output_dir)
                all_em_results[folder_name] = em_results
                print(f"Processed folder: {folder_name}")
            except Exception as e:
                print(f"Error processing folder {folder_name}: {str(e)}")
    return all_em_results

def create_dice_table(em_results):
    rows = []
    for folder, results in em_results.items():
        dice_values = [results['dice_scores'][label] for label in range(1, 4)]
        rows.append({
            'Folder': folder,
            'CSF Dice': results['dice_scores'][1],
            'GM Dice': results['dice_scores'][2],
            'WM Dice': results['dice_scores'][3],
            'Mean Dice': np.mean(dice_values),
            'Std Dice': np.std(dice_values)
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parent_directory = '/path/to/input/folders'
    output_parent_directory = '/path/to/output/folders'
    all_em_results = process_all_folders_with_em(parent_directory, output_parent_directory)
    dice_table = create_dice_table(all_em_results)
    print(dice_table)
    dice_table.to_csv(os.path.join(output_parent_directory, "dice_scores_summary.csv"), index=False)
