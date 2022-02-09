python run_translation.py \
    --output_dir 'outputs' \
    --train_file 'data/train.json' \
    --validation_file 'data/validation.json' \
    --model_name_or_path 'models/initial-mdeberta-to-gpt2' \
    --source_lang 'src' \
    --target_lang 'tgt' \
    --overwrite_output_dir \
    --do_train \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --logging_steps 1000 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0