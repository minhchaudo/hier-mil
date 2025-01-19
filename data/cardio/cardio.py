# Download the following files from https://singlecell.broadinstitute.org/single_cell/study/SCP1303/
# 1. DCM_HCM_Expression_Matrix_raw_counts_V1.mtx
# 2. DCM_HCM_Expression_Matrix_genes_V1.tsv
# 3. DCM_HCM_Expression_Matrix_barcodes_V1.tsv
# 4. DCM_HCM_MetaData_V1.txt


from scipy.io import mmread
import scanpy as sc
import pandas as pd

data = mmread("DCM_HCM_Expression_Matrix_raw_counts_V1.mtx")

genes = pd.read_csv("DCM_HCM_Expression_Matrix_genes_V1.tsv", sep="\t", header=None).iloc[:,1].tolist()

barcodes = open( "DCM_HCM_Expression_Matrix_barcodes_V1.tsv").read().strip().split("\n")

meta = pd.read_csv("DCM_HCM_MetaData_V1.txt", sep="\t").drop(axis=0,index=0).reset_index(drop=True)

adata = sc.AnnData(data)
adata.obs.index = barcodes
adata.var.index = genes

sc.pp.filter_genes(adata, min_cells=5)

sc.pp.normalize_total(adata, target_sum=1e4)

sc.pp.log1p(adata)

meta.set_index("NAME", inplace=True)
adata.obs = meta.loc[adata.obs.index, :]
adata.obs["label"] = adata.obs["disease__ontology_label"].apply(lambda x: 0 if x=="normal" else 1 if x=="hypertrophic cardiomyopathy" else 2)
adata.obs.rename({"donor_id":"patient", "cell_type__ontology_label":"cell_type_annotation"}, inplace=True)

# Extract the cell embeddings using the scGPT model (scgpt.tasks.embed_data) with the pretrained weights from the whole-human checkpoint. See https://github.com/bowang-lab/scGPT for instructions. 
# Store the embeddings in a new adata object and include the metadata of the old adata. Write the new adata object to the file cardio.h5ad .
#

