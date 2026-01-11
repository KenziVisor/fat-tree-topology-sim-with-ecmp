#!/bin/bash

echo "=============================================="
echo " Running Classic ECMP Experiment Suite"
echo "=============================================="

set -e
PYTHON=python3
SCRIPT="fat-tree-topology-sim.py"

# ------------------------------------------------
# Scenario A+B with moderate load
# ------------------------------------------------
echo ""
echo ">>> ECMP classic: k=4 , 2000 flows"
$PYTHON $SCRIPT --experiment ecmp -k 4 --num_flows 2000

# ------------------------------------------------
# Higher load to amplify ECMP imbalance
# ------------------------------------------------
echo ""
echo ">>> ECMP classic: k=4 , 4000 flows (stress test)"
$PYTHON $SCRIPT --experiment ecmp -k 4 --num_flows 4000

# ------------------------------------------------
# Larger topology to show ECMP improves with scale
# ------------------------------------------------
echo ""
echo ">>> ECMP classic: k=8 , 4000 flows"
$PYTHON $SCRIPT --experiment ecmp -k 8 --num_flows 4000

echo ""
echo "=============================================="
echo " Classic ECMP experiments completed!"
echo "=============================================="
