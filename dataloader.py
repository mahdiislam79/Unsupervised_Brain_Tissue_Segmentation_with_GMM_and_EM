import nibabel as nib
import numpy as np

def load_mri_image(image_path):
    """Load an MRI image from the given file path."""
    img_data = nib.load(image_path)
    return img_data.get_fdata()

def get_data_within_labels(image_data, labels):
    """Extract image data corresponding to labeled regions."""
    return image_data[labels > 0]

def prepare_data_for_clustering(t1_image_data, labels):
    """Prepare data for clustering by extracting relevant voxels."""
    y_data = get_data_within_labels(t1_image_data, labels)
    return np.array([y_data]).T
