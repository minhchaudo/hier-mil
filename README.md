# Hierarchical MIL

Code for the manuscript: Incorporating Hierarchical Information into Multiple
Instance Learning for Patient Phenotype Prediction
with scRNA-seq Data

## Setup

```
git clone https://github.com/minhchaudo/hier-mil
cd hier-mil
python -m venv venv
source hier-mil/bin/activate
pip install -r requirements.txt
```

## Training and Inference

To train a model, our implementation expects a `.h5ad` file readable by `Scanpy` as input. The `.X` attribute of the `adata` object should contain either the gene expression or the latent representations of all single cells in the dataset. The `.obs` attribute of the `adata` object should contain at least the patient identifiers, the labels (encoded as 0/1 for binary tasks or as integers for multiclass tasks), and the cell type annotations.

For training and tuning a model, run

```
python run.py \
--data_path [input_path] \
--model_save_path [model_path] \
--task 0 \
--patient_id_key [patient_id_key] \
--label_key [label_key]
--cell_type_annot_key [cell_type_annot_key] \
--attn1 [attn1]

```

where `[input_path]` is the path to the input file, `[model_path]` is the path to save the trained model, `[patient_id_key]` is the column name of the patient IDs in `.obs`, `[label_key]` is the column name of the labels, and `[cell_type_annot_key]` is the column name of the cell type annotations. Please set the option `[attn1]` to `0` to use the CTA model and to `1` (the default) to use the HA model.

To predict on new data, run

```
python run.py \
--data_path [input_path] \
--model_save_path [model_path] \
--task 1 \
--patient_id_key [patient_id_key] \
--cell_type_annot_key [cell_type_annot_key] \
--attn1 [attn1]

```

Note that the number of input dimensions (the number of dimensions in the `adata` object) should be the same for both training and inference data.

## Exeperiments

To reproduce the experiments described in the manuscript:

- Download and preprocess the four datasets following Section 4.1 in the manuscript
- Name the files `cardio.h5ad`, `covid.h5ad`, `aml-icb.h5ad`, and `icb.h5ad` accordingly and put them in the same directory as the code
- Run `sh run.sh`
