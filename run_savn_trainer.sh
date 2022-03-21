# Experiement with 5 speeds for SAVN with original setting

dataset_version="2.1"
max_train_samples=-1
num_train_epochs=5
use_sp=true
date=$(date +%d-%m-%Y" "%H:%M:%S)
tensorboard_name="runs/SA_${version}.${date}"
utils_version="SA_${version}"

# use_sp
if [ "$use_sp" = true ]; then
	version="sp_$dataset_version"
else
	version="raw_${dataset_version}"
fi

utils_version="SA_${version}"


#  32 64 100 300 500 800 1000
for seed in 10 15 32 42 1074
do
    # output_dir
    if [ $max_train_samples -gt 1 ]; then
        output_dir="output/freeze_savn_SA_${version}_seed-${seed}_${max_train_samples}"
    else
        output_dir="output/freeze_savn_SA_${version}_seed-${seed}"
    fi
    mkdir -p $output_dir
    echo $output_dir
    python run_savn_trainer.py \
        --model_name_or_path "bert-base-uncased" \
        --tokenizer_name "model/bert-base-savn-vocab.txt" \
        --config_name "model/bert-base-uncased-config.json" \
        --dataset_version 2.1 \
        --use_sp \
        --use_vn \
        --tensorboard_name "${tensorboard_name}" \
        --utils_version "${utils_version}" \
        --per_gpu_eval_batch_size 16 \
        --per_gpu_train_batch_size 8 \
        --eval_steps 0 \
        --save_steps 0 \
        --evaluate_during_training \
        --train_file "data/${dataset_version}/train_dials.json" \
        --validation_file "data/${dataset_version}/dev_dials.json" \
        --predict_file "data/${dataset_version}/test_dials.json" \
        --output_dir "$output_dir" \
        --do_train True \
        --do_eval True \
        --do_predict True \
        --num_train_epochs $num_train_epochs \
        --evaluate_during_training \
        --seed $seed \
        --freeze_model True \
        --max_train_samples $max_train_samples \
        --overwrite_output_dir \
        --fp16 \
    	2>&1 | tee ${output_dir}/sa_train.log
done

