# Download the following files from https://singlecell.broadinstitute.org/single_cell/study/SCP1289/ :
# 1. 20210220_NasalSwab_RawCounts.txt
# 3. 20210220_NasalSwab_NormCounts.txt
# 2. 20210701_NasalSwab_MetaData.txt

import pandas as pd
import scanpy as sc

df = pd.read_csv("20210220_NasalSwab_RawCounts.txt", sep='\t')

adata = sc.AnnData(df.T)

adata.obs.index = df.columns
adata.var.index = df.index

sc.pp.filter_genes(adata, min_cells=5)

sc.pp.normalize_total(adata, target_sum=1e4)

sc.pp.log1p(adata)

meta = pd.read_csv("20210701_NasalSwab_MetaData.txt", sep="\t").drop(axis=0,index=0).reset_index(drop=True)

meta.set_index("NAME", inplace=True)

adata.obs = meta.loc[adata.obs.index, :]

adata.obs["label"] = adata.obs["disease__ontology_label"].apply(lambda x: 0 if x=="normal" else 1 if x=="COVID-19" else -1)

adata = adata[adata.obs["label"] != -1]

adata.obs.rename({"donor_id":"patient", "Coarse_Cell_Annotations": "original_cell_type_annotation"}, inplace=True)

adata.write_h5ad("covid.h5ad")

# Extract the cell embeddings using the scGPT model (scg.tasks.embed_data) with the pretrained weights from the whole-human checkpoint. See https://github.com/bowang-lab/scGPT for instructions. 
# Store the embeddings in a new adata object and include the metadata of the old adata. Write the new adata object to the file covid.h5ad .
#

adata = sc.read_h5ad("covid.h5ad")

ct = pd.read_csv("singler_covid.csv", index_col=0)

adata.obs["cell_type_annotation"] = ct.loc[adata.obs.index, "pruned.labels"]

adata = adata[adata.obs["cell_type_annotation"].notna()]

adata.write_h5ad("../covid.h5ad")








