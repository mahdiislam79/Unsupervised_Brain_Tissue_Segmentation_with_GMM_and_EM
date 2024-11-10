import numpy as np

def dice_score(segmentation, ground_truth, label):
    """Calculate the Dice score for a given label."""
    intersection = np.sum((segmentation == label) & (ground_truth == label))
    union = np.sum(segmentation == label) + np.sum(ground_truth == label)
    return 2 * intersection / union if union > 0 else 1.0

def calculate_dice_scores(segmentation, ground_truth, labels):
    """Calculate Dice scores for all specified labels."""
    dice_scores = {}
    for label in labels:
        dice_scores[label] = dice_score(segmentation, ground_truth, label)
    return dice_scores
