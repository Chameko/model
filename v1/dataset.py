import os
from torch.utils.data import Dataset
import torch
import scanpy as sc
import tifffile
from scanpy import read_visium
import pandas as pd
import numpy as np

Breast_Cancer = [
    "V1_Breast_Cancer_Block_A_Section_1"
    "V1_Breast_Cancer_Block_A_Section_2"
    "Targeted_Visium_Human_BreastCancer_Immunology"
]
data_dir = "../data"

class TxGBreastDataset(Dataset):
    """Contains data various from 10x Genomics breast cancer spatial transcriptomic data and their corresponding
    slide images

    breast_cancers -- Slides to include in req_gene
    window_size  -- Size  of the window
    all_data -- Contains the gene expression data, indexed by slide
    req_gene -- Contains the gene expression of the slides specified by breast_cancers
    all_gene -- Contains the gene expression of all slides
    mean -- Contains the mean gene expression of all slides
    max -- Contains the maximum gene expression of slides specified by breast_cancers
    min -- Contains the minimum gene expression of slides specified by breast_cancers"""
    
    def __init__(self, data_dir, breast_cancers, window_size):
        """Init the dataset
        data_dir -- The directory containing the data
        breast_cancers -- Which breast cancer datasets to include
        window_size -- Size of the windows"""
        self.breast_cancers = breast_cancers
        self.window_size = window_size
        data = []
        for slide in Breast_Cancer:
            # Get the path
            path = os.path.join(data_dir, slide)
            # Get the data
            data.append(self.load_data(path))
        
        # Process the data
        self.process_data(data)
        # We keep the top 250 genes with the highest mean
        # The mean with the current index stored in the second half
        mean_with_idx = zip(self.mean, range(self.mean.shape[0]))
        top_250 = sorted(mean_with_idx, reverse=True)[:250][1]
        # Record the gene names of the filtered genes
        self.filter_name = []
        for (i, name) in enumerate(self.gene_names):
            if i in top_250 : self.filter_name.append(name)
        # Boolean filter for the genes
        self.gene_filter = []
        for i in range(len(self.gene_names)):
            self.gene_filter.append(i in top_250)
        self.gene_filter = np.array(self.gene_filter)

        # Update the maximum and minimum. Log transformed with +1 to prevent log(0)
        self.max = torch.log10(torch.as_tensor(self.max[self.gene_filter], dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min[self.gene_filter], dtype=torch.float) + 1)
        
        # Create flat mapping of index -> gene count of slide
        self.mapping = []
        for i in self.breast_cancers:
            _, counts = self.all_data[i]
            for j in range(len(counts)):
                self.mapping.append([i, j])

    def load_data(self, data_root):
        """Load the data from a directory
        data_root -- directory to load data from"""
        
        # Get the name of the dataset from the path
        dataset_name = data_root.split(os.sep)[-1]
        # Get the count file 
        count_file = os.path.join(data_root, f"{dataset_name}_filtered_feature_bc_matrix.h5")
        # Load the data from the directory.
        data = read_visium(data_root, count_file=count_file)
        data.var_names_make_unique()
        # Mark mitodrial genes
        data.var['mt'] = data.var_names.str.startswith('MT-')
        # Load the image
        img = tifffile.tifffile.imread(os.path.join(data_root, f"{dataset_name}_image.tif"))
        # QC metrics
        sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], inplace=True)
        return [img, data]

    def process_data(self, data):
        """Processes the dataset"""
        
        from tqdm import tqdm
        gene_names = set()
        # We get all the gene names present in the dataset
        for (img, d) in tqdm(data, "gathering gene names"):
            # Get the counts of genes
            counts = pd.DataFrame(d.X.todence(), columns=d.var_names, index=d.obs_names)
            # Get the coords of genes
            coord = pd.DataFrame(d.obsm['spatial'], columns=['x_coord', 'y_coord'], index=d.obs_names)
            # Record gene names
            gene_names = gene_names.union(set(counts.columns.values))
        all_data = []
        all_gene = []
        req_gene = []
        gene_names = list(gene_names)
        gene_names.sort()
        # We get the gene counts and pad any missing genes with 0s
        for (idx, (img, d)) in tqdm(enumerate(data), "padding data"):
            counts = pd.DataFrame(d.X.todence(), columns=d.var_names, index=d.obs_names)
            coord = pd.DataFrame(d.obsm['spatial'], columns=['x_coord', 'y_coord'], index=d.obs_names)
            # Record the missing genes
            missing = list(set(gene_names) - set(counts.columns.values))
            padded_counts = counts.values.astype(float)
            # Create zeroed values
            padding = np.zeros((padded_counts.shape[0], len(missing)))
            padded_counts = np.concatenate((padded_counts, padding), 1)
            # Sort the counts
            names = np.concatenate((counts.columns.values, np.array(missing)))
            padded_counts = padded_counts[:, np.argsort(names)]
            
            # Gather all the data
            all_data[idx] = [img, padded_counts, coord.values.astype(int)]
            # Gather all the genes
            for gene in padded_counts:
                all_gene.append(gene)
                if idx in self.breast_cancers:
                    req_gene.append(gene)

        all_gene = np.array(all_gene)
        req_gene = np.array(req_gene)

        self.mean = np.mean(all_gene, 0)
        self.max = np.max(req_gene, 0)
        self.min = np.min(req_gene, 0)
        self.gene_names = gene_names
        self.all_data = all_data

    def __getitem__(self, index):
        # Get the correct slide and gene count based on mapping index
        slide_idx, count_idx = self.mapping[index]
        img, counts, coords = self.all_data[slide_idx]
        count = counts[count_idx]
        x, y = coords[count_idx]
        # Get the position in terms of windows
        position = [x // self.window_size, y // self.window_size]

        # Get the image pixels, centered on the window position
        img = img[(y - (self.window_size // 2)):(y + (self.window_size // 2)), (x - (self.window_size // 2)):(x + (self.window_size // 2))]
        img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1)
        # Log transform. 1 added so we don't log(0)
        counts = torch.log10(torch.as_tensor(counts[self.gene_filter], dtype=torch.float) + 1)
        # Min-max normalize
        counts = (counts - self.min) / (self.max - self.min)

        return {
            "img" : img,
            "count" : count,
            "pos" : torch.LongTensor(position)
        }
