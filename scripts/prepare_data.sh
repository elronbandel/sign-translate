python prepare_data.py \
    --source_file 'data/data.csv.gz' \
    --target_file 'data/train.json' \
    --index_col 0 \
    --wrapper_col 'translation' \
    --end_index 25000 

python prepare_data.py \
    --source_file 'data/data.csv.gz' \
    --target_file 'data/validation.json' \
    --index_col 0 \
    --wrapper_col 'translation' \
    --start_index 25000 