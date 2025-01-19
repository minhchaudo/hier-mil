#!/bin/bash
datasets=("cardio" "covid" "icb")

# repeated 10-fold CV

for dataset in "${datasets[@]}"
do
    for attn1 in 0 1 
    do
        filename="${dataset}.h5ad"
        output="${dataset}_attn1_${attn1}_attn2_1_repeated_k_fold_out.csv"  

        python run.py \
        --data_path "$filename" \
        --output "$output" \
        --task 2 \
        --n_repeats 10 \
        --n_folds 10 \
        --n_folds_hyperparam_tune 10 \
        --attn1 "$attn1" \
        --attn2 1
    done
done

# vary train size

for dataset in "${datasets[@]}"
do
    # Task 3: Vary train size
    for attn1 in 0 1
    do
        filename="${dataset}.h5ad"
        output="${dataset}_attn1_${attn1}_attn2_1_vary_train_size_out.csv" 
        python run.py \
        --data_path "$filename" \
        --output "$output" \
        --task 3 \
        --n_repeats 100 \
        --n_folds_hyperparam_tune 4 \
        --attn1 "$attn1" \
        --attn2 1
    done
done

# vary cell count

for dataset in "${datasets[@]}"
do
    for attn1 in 0 1
    do
        filename="${dataset}.h5ad"
        output="${dataset}_attn1_${attn1}_attn2_1_vary_cell_count_out.csv"

        python run.py \
        --data_path "$filename" \
        --output "$output" \
        --task 4 \
        --n_repeats 10 \
        --n_folds 10 \
        --n_folds_hyperparam_tune 10 \
        --attn1 "$attn1"\
        --attn2 1
    done
done

# randomize cell type annot

for dataset in "${datasets[@]}"
do
    for attn1 in 0 1
    do
        filename="${dataset}.h5ad"
        output="${dataset}_attn1_${attn1}_attn2_1_random_ct_annot_out.csv"

        python run.py \
        --data_path "$filename" \
        --output "$output" \
        --task 5 \
        --n_repeats 10 \
        --n_folds 10 \
        --n_folds_hyperparam_tune 10 \
        --attn1 "$attn1"\
        --attn2 1
    done
done

# ablation
for dataset in "${datasets[@]}"
do
    for attn1 in 0 1 
    do
        filename="${dataset}.h5ad"
        output="${dataset}_attn1_${attn1}_attn2_0_repeated_k_fold_out.csv"  

        python run.py \
        --data_path "$filename" \
        --output "$output" \
        --task 2 \
        --n_repeats 10 \
        --n_folds 10 \
        --n_folds_hyperparam_tune 10 \
        --attn1 "$attn1"  \
        --attn2 0 
    done
done

# identify key cell types covid

filename="covid.h5ad"
output="covid_key_ct_out.csv"

python run.py \
--data_path "$filename" \
--output "$output" \
--task 6 \
--attn1 1 \
--attn2 1 \
--n_perm 100 \
--n_folds 5 \
--n_epochs 1000 \
--dropout 0.5 \
--weight_decay 0.001 



