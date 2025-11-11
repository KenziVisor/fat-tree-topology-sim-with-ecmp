# Fat-Tree Topology Simulation

This project builds and analyzes a **k-ary Fat-Tree data center network topology**.  
The simulation supports two main capabilities:

1. **Drawing the network topology**
2. **Simulating random link failures and analyzing performance effects**

Fat-Tree topologies are widely used in real data centers because they provide **multiple redundant paths**, improving **fault tolerance**, **throughput**, and **scalability**.

---

\## ⚙️ Installation

```bash

git clone git@github.com:KenziVisor/fat-tree-topology-sim.git
cd fat-tree-topology-sim
python3 -m venv .venv
source .venv/bin/activate
pip install networkx matplotlib numpy
```
---

## Usage

### Draw the topology (no link failures)

    python fat-tree-topology-sim.py -k 4

### Draw the topology and apply link failures

    python fat-tree-topology-sim.py -k 4 -p 0.1

### Run a performance experiment over multiple failure probabilities

    python fat-tree-topology-sim.py -k 4 --sweep "0,0.1,0.2,0.3,0.5" -t 10

| Argument | Meaning | Example |
|---------|---------|---------|
| `-k` | Fat-tree size (must be a positive even number) | `-k 4` |
| `-p` / `--prob` | Single link failure probability (0–1) | `-p 0.1` |
| `--sweep` | Multiple probabilities to test | `--sweep "0,0.2,0.4"` |
| `-t` / `--trials` | Number of averaging runs per probability | `-t 10` |

---

## What the Program Does

A **k-ary Fat-Tree** is constructed with four layers:

Core Layer
Aggregation Layer
Edge Layer
Host Layer


The program can:

- **Draw the full topology** showing each layer.
- Create **random host-to-host communication pairs**.
- Apply **switch-to-switch random link failures** with probability `p`.
- Measure:
  - **Average shortest-path length**
  - **Reachable host-pair fraction** (connectivity)

---

## Interpreting the Graphs

Two curves are plotted:

- **Blue solid line** — Average shortest-path length  
  (How much longer paths become under failures)

- **Red dashed line** — Reachable pair fraction  
  (How many host pairs still have a working route)

### Observations

- When `p = 0`, connectivity is full and path lengths are minimal.
- As `p` increases:
  - Some links fail.
  - Paths become slightly longer due to detours.
  - Beyond a threshold, some host pairs become disconnected.
- Larger **k** → More redundant paths → More resilience.

### Conclusion

**Fat-Tree networks degrade gracefully under link failures.**  
Even when links fail, most hosts remain connected, and routing still operates efficiently, especially at larger scales.  
This property explains their widespread use in real-world data centers.

---

## Output Files

| File | Description |
|------|-------------|
| `fat tree topology k = X p = 0.png` | Visualization of the network structure |
| `average lengths as a function of link failure probability k = X trials = Y.png` | Performance graph |

---

End of README.