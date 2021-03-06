PYTHON=venv/bin/python

# Generate small
mkdir -p dataset
$PYTHON generate.py --layer_size_start 32 --layer_size_stop 257 \
    --layer_size_step 32 --layer_count_start 1 --layer_count_stop 3 \
    --layer_count_step 1 --out_dir dataset
