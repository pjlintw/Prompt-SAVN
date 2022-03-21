# Experiement with 5 speeds for prompt-SAVN and 

dataset_version="2.1"
max_train_samples=-1
num_train_epochs=5
use_sp=true
date=$(date +%d-%m-%Y" "%H:%M:%S)
tensorboard_name="runs/SA_${version}.${date}"


# use_sp
if [ "$use_sp" = true ]; then
	version="sp_$dataset_version"
else
	version="raw_${dataset_version}"
fi
utils_version="SA_${version}"

# 32 64 100 300 500 800 1000 3000 5000
for seed in 10 15 32 42 1074
do  
    # output_dir
    if [ $max_train_samples -gt 1 ]; then
        output_dir="output/freeze_prompt-savn_SA_${version}_seed-${seed}_${max_train_samples}"
    else
        output_dir="output/freeze_prompt-savn_SA_${version}_seed-${seed}"
    fi
    mkdir -p $output_dir
    echo $output_dir
    python run_savn_trainer.py \
        --model_name_or_path "bert-base-uncased" \
        --tokenizer_name "model/bert-base-savn-vocab.txt" \
        --config_name "model/bert-base-uncased-config.json" \
        --dataset_version 2.1 \
        --train_file "data/${dataset_version}/train_dials.json" \
        --validation_file "data/${dataset_version}/dev_dials.json" \
        --predict_file "data/${dataset_version}/test_dials.json" \
        --use_sp \
        --use_vn \
        --output_dir "$output_dir" \
        --tensorboard_name "${tensorboard_name}" \
        --utils_version "${utils_version}" \
        --do_train True \
        --do_eval True \
        --do_predict True \
        --num_train_epochs $num_train_epochs \
        --per_gpu_eval_batch_size 16 \
        --per_gpu_train_batch_size 8 \
        --eval_steps 0 \
        --save_steps 0 \
        --evaluate_during_training \
        --seed $seed \
        --max_train_samples $max_train_samples \
        --use_pattern True \
        --overwrite_output_dir \
        --fp16 \
    	2>&1 | tee ${output_dir}/sa_train.log
done