import numpy as np
from data_loader import load_mri_image, prepare_data_for_clustering
from clustering import initialize_clusters, calculate_sigma, em_algorithm
from evaluation import calculate_dice_scores

def run_pipeline(t1_image_path, gt_path, labels=[1, 2, 3]):
    # Load MRI data and ground truth
    t1_image = load_mri_image(t1_image_path)
    ground_truth = load_mri_image(gt_path)
    
    # Prepare data for clustering
    data = prepare_data_for_clustering(t1_image, ground_truth)
    
    # Initialize clusters
    mu, labels = initialize_clusters(data)
    sigma = calculate_sigma(data, labels)
    pi = [1/3] * 3  # Assuming equal priors for simplicity
    
    # Run EM algorithm
    mu, sigma, pi, responsibilities = em_algorithm(data, mu, sigma, pi)
    
    # Assign labels based on the clustering outcome
    sorted_indices = np.argsort(mu[:, 0])  # Sort clusters by mean intensity
    label_mapping = {sorted_indices[i]: i+1 for i in range(3)}
    assigned_labels = np.vectorize(label_mapping.get)(np.argmax(responsibilities, axis=1))
    
    # Create segmented image from labels
    segmentation = np.zeros_like(ground_truth)
    segmentation[ground_truth > 0] = assigned_labels.flatten()
    
    # Evaluate segmentation using Dice scores
    dice_scores = calculate_dice_scores(segmentation, ground_truth, labels)
    
    # Print Dice scores
    for label, score in dice_scores.items():
        print(f"Dice score for label {label}: {score}")
    
    return segmentation, dice_scores

# Example usage
if __name__ == "__main__":
    # Set file paths (adjust these paths based on your environment)
    t1_image_path = 'path/to/T1.nii'
    gt_path = 'path/to/LabelsForTesting.nii'
    
    # Run the pipeline
    segmentation, dice_scores = run_pipeline(t1_image_path, gt_path)
