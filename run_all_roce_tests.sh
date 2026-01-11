#!/bin/bash

# ============================================================
# RoCE Flowlet Experiment Full Test Suite
# File: run_all_roce_tests.sh
# ============================================================

echo "=============================================="
echo " Running Full RoCE Experiment Test Suite"
echo "=============================================="

# Stop script on first error
set -e

PYTHON=python3
SCRIPT="fat-tree-topology-sim.py"

# -----------------------------
# Test 1 — Basic RoCE sanity run
# -----------------------------
echo ""
echo ">>> Test 1: Basic RoCE sanity run"
$PYTHON $SCRIPT --experiment roce -k 4 --num_flows 2000 --qps 8 --trials 20

# -----------------------------
# Test 2 — Sweep QPs (e-ECMP effect)
# -----------------------------
echo ""
echo ">>> Test 2: QPs sweep"
$PYTHON $SCRIPT --experiment roce -k 8 --num_flows 2000 --qps_sweep 1,2,4,8,16 --trials 20

# -----------------------------
# Test 3 — Sweep trials (Flowlet effect)
# -----------------------------
echo ""
echo ">>> Test 3: Trials sweep"
$PYTHON $SCRIPT --experiment roce -k 8 --num_flows 2000 --qps 8 --trials_sweep 1,2,5,10,20,40

# -----------------------------
# Test 4 — Sweep k (Topology scaling)
# -----------------------------
echo ""
echo ">>> Test 4: k sweep"
$PYTHON $SCRIPT --experiment roce --k_sweep 4,6,8,16,32 --num_flows 2000 --qps 8 --trials 20

# -----------------------------
# Optional — Strong visual results
# -----------------------------
echo ""
echo ">>> Optional Test: High-load showcase run"
$PYTHON $SCRIPT --experiment roce -k 10 --num_flows 4000 --qps_sweep 1,2,4,8,16,32 --trials 30

echo ""
echo "=============================================="
echo " All RoCE experiments completed successfully!"
echo " Check generated .png files in current folder."
echo "=============================================="
