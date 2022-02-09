python prepare_model.py \
    --source_model 'microsoft/mdeberta-v3-base' \
    --target_model 'gpt2' \
    --final_name 'initial-mdeberta-to-gpt2' \
    --outputs_dir 'models' \
    --vocab_file 'data/vocab.txt' 