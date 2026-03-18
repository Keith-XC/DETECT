#!/bin/bash

# set parameters
TASK="facial"          # : facial, dog, yolo
MODEL="small"         # for facial model small:resnet   large:VIT
START_SEED=0
END_SEED=50           #
EXTENT=10
CONFIDENCE_THRESHOLD=-10
ORACLE="confidence_drop"

# define all configs
CONFIGS=("gradient" "smoothgrad" "random" "occlusion" ) # "gradient"  "occlusion" "smoothgrad"

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