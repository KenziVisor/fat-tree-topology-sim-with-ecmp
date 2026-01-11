#!/bin/bash

echo "=============================================="
echo " Running Fat-Tree Resilience Experiment Suite"
echo "=============================================="

set -e
PYTHON=python3
SCRIPT="fat-tree-topology-sim.py"

# Higher failure probabilities (more interesting curves)
SWEEP="0,0.05,0.1,0.2,0.3,0.5"
TRIALS=10

echo ""
echo ">>> Fat-tree resilience: k=4"
$PYTHON $SCRIPT --experiment fat_tree -k 4 --sweep $SWEEP --trials $TRIALS

echo ""
echo ">>> Fat-tree resilience: k=8"
$PYTHON $SCRIPT --experiment fat_tree -k 8 --sweep $SWEEP --trials $TRIALS

# Optional: k=16 might be heavy depending on your machine
echo ""
echo ">>> Fat-tree resilience: k=16 (optional, may take longer)"
$PYTHON $SCRIPT --experiment fat_tree -k 16 --sweep $SWEEP --trials $TRIALS

echo ""
echo "=============================================="
echo " Fat-Tree resilience experiments completed!"
echo "=============================================="
