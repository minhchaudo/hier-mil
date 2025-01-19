# Download Rshinydata_singlecell-20231219T155916Z-001.zip from https://zenodo.org/records/10407126 and unzip the folder

import pandas as pd
import os
import scanpy as sc

dfs = []
for f in os.listdir("Rshinydata_singlecell"):
    df = pd.read_csv("Rshinydata_singlecell/"+f)
    dfs.append(df)

df = pd.concat(dfs, axis=0)
df = df.iloc[:, 1:]
df.set_index("Row.names", inplace=True)

meta = df.iloc[:,:195]
X = df.iloc[:,195:]

meta = meta[meta["sample_id_pre_post"].apply(lambda x: x.split("_")[-1] == "Pre")]
meta["label"] = meta["Combined_outcome"].apply(lambda x: 0 if x=="Unfavourable" else 1 if x=="Favourable" else -1)
meta = meta[meta["label"] != -1]

sample_to_patient = pd.read_csv("icb_sample_id_to_patient_id.csv").set_index("sample_id")
meta["patient"] = meta["sample_id_pre_post"].apply(lambda x: sample_to_patient.loc[x, "patient"])

ct = pd.read_csv("singler_icb_pre.csv", index_col=0)
meta["cell_type_annotation"] = ct["pruned.labels"]
meta = meta[meta["cell_type_annotation"].notna()]

genes = pd.read_csv("genes_icb.csv")["Gene"].to_list()
adata = sc.AnnData(X[genes])
adata.obs.index = X.index
adata.var.index = genes
adata.obs = meta.loc[adata.obs.index, :]

adata.write_h5ad("../icb.h5ad")


