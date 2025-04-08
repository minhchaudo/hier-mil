import scanpy as sc
import argparse
from train import repeated_k_fold, train_and_tune, predict_and_save
from run_permute import get_p_val_cell_type
from vary_data_quality import vary_cell_count, vary_train_size, randomize_cell_annot
from utils import get_df, get_meta
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--meta_cols", default=None)
    parser.add_argument("--output", default="out.txt")
    parser.add_argument("--model_save_path", default="model.pt")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--patient_id_key", default="patient")
    parser.add_argument("--label_key", default="label")
    parser.add_argument("--cell_type_annot_key", default="cell_type_annotation")
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_folds_hyperparam_tune", type=int, default=10)
    parser.add_argument("--n_perm", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn1", type=int, default=1)
    parser.add_argument("--attn2", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--n_layers_lin", type=int, default=1)
    parser.add_argument("--n_layers_lin_meta", type=int, default=1)
    parser.add_argument("--n_hid", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_tune_trials", type=int, default=30)

    args = parser.parse_args()

    adata = sc.read_h5ad(args.data_path)

    args.use_meta = True if args.meta_cols is not None else False
    args.n_classes = len(set(adata.obs[args.label_key]))
    args.binary = args.n_classes == 2
    args.all_ct = adata.obs[args.cell_type_annot_key].unique()

    df = get_df(adata, args.patient_id_key, args.label_key, args.cell_type_annot_key, no_label=(args.task==1))

    if args.use_meta:
        with open(args.meta_cols, "r") as file:
            meta_cols = [line.strip() for line in file]
            meta = get_meta(adata, meta_cols)

    else:
        meta=None

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUDNN_DETERMINISTIC"] = "1"
    
    # train and tune
    if args.task == 0:
        model = train_and_tune(df, meta, args)
        print(f"Model saved to {args.model_save_path}")

    # predict
    if args.task == 1:
        pred = predict_and_save(df, meta, args)
        print(f"Predictions saved to {args.output}")

    # repeated k fold
    if args.task == 2:
        mean_auc, std_auc = repeated_k_fold(df, meta, args)
        print(f"AUC: {mean_auc} +/- {std_auc}")
        print(f"Results saved to {args.output}")

    # vary train size
    if args.task == 3:
        train_sizes = [0.25, 0.5, 0.75]
        res = vary_train_size(train_sizes, df, args)
        print(f"Results saved to {args.output}")

    # vary cell count
    if args.task == 4:
        cell_counts = [0.25, 0.5, 0.75]
        res = vary_cell_count(cell_counts, df, args)
        print(f"Results saved to {args.output}")

    # randomize cell annot
    if args.task == 5:
        cell_props = [0.25, 0.5]
        res = randomize_cell_annot(cell_props, df, args)
        print(f"Results saved to {args.output}")

    # permute
    if args.task == 6:
        res = get_p_val_cell_type(df, args)
        print(f"Results saved to {args.output}")
        

