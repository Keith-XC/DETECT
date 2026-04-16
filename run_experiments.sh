#!/bin/bash
#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export CUDA_VISIBLE_DEVICES=0

# set parameters
TASK="yolo"          # : facial, dog, yolo
MODEL="small"         # for facial model small:resnet   large:VIT
START_SEED=0
END_SEED=10           #
EXTENT=10
CONFIDENCE_THRESHOLD=0.4
ORACLE="confidence_drop"

# define all configs
CONFIGS=("smoothgrad") # "random" "gradient" "smoothgrad" "occlusion"

echo "Starting automated runs for task: $TASK"
echo "------------------------------------------"

for CFG in "${CONFIGS[@]}"
do
    echo "==== Currently running config: $CFG ===="

    python main.py \
        --task "$TASK" \
        --model "$MODEL" \
        --config "$CFG" \
        --oracle "$ORACLE" \
        --extent_factor "$EXTENT" \
        --start_seed "$START_SEED" \
        --end_seed "$END_SEED" \
        --confidence_threshold "$CONFIDENCE_THRESHOLD"

    echo "Finished $CFG."
    echo "------------------------------------------"
done

echo "All configs have been processed!"