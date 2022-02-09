python prepare_data.py \
    --source_file 'data/data.csv.gz' \
    --target_file 'data/train.json' \
    --index_col 0 \
    --wrapper_col 'translation' \
    --end_percent 90

python prepare_data.py \
    --source_file 'data/data.csv.gz' \
    --target_file 'data/validation.json' \
    --index_col 0 \
    --wrapper_col 'translation' \
    --start_percent 90