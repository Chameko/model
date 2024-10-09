from scanpy import read_visium
import scanpy as sc
import os
import tifffile

def load_raw(dir):
    """Loads the raw data from a directory
    dir -- The directory to load the data from"""
    # Get the dataset name as the name of the dir
    dataset_name = dir.split(os.sep)[-1]
    # Load data
    data = read_visium(dir, f"{dataset_name}_filtered_feature_bc_matrix.h5")
    data.var_names_make_unique()
    # Load image
    img = tifffile.imread(os.path.join(dir, f"{dataset_name}_image.tif"))
    # Mark mitocondrial genes
    data.var["mt"] = data.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], inplace=True)
    
def preprocess():
    # TODO
