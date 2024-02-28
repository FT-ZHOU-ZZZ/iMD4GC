#!/bin/sh
models="iMD4GC"
for model in $models
do
  CUDA_VISIBLE_DEVICES=$num python main.py --model $model \
                                          --excel_file /data/GastricCancer/TCGA-STAD/All_Statistics.xls \
                                          --modal Clinical_WSI_Omics \
                                          --num_epoch 25 \
                                          --batch_size 1 \
                                          --n_features 768 \
                                          --fusion ConcatWithLinear \
                                          --layers 2 \
                                          --loss nll_surv_kd_kd \
                                          --alpha 8.0 \
                                          --beta 0.0 
done 