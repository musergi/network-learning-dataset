PYTHON=venv/bin/python
mkdir -p dataset
$PYTHON generate.py --layer_size_start 8 --layer_size_stop 32 \
    --layer_size_step 8 --layer_count_start 1 --layer_count_stop 3 \
    --layer_count_step 1 --out_dir dataset
