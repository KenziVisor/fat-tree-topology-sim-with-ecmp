# Fat-Tree Routing Simulation: ECMP, e-ECMP, and Flowlet-Based Load Balancing

This project implements a Python-based simulator for studying routing behavior in fat-tree data center networks. 
The simulator focuses on multipath routing mechanisms used in modern large-scale clusters, including Equal-Cost Multi-Path (ECMP), enhanced ECMP using multiple Queue Pairs (e-ECMP), and flowlet-based path reshuffling.

Rather than modeling packet-level transport or queuing, the simulator isolates routing decisions at the flow and flowlet level. 
This allows direct examination of how different hashing-based routing schemes distribute traffic load across network links and mitigate congestion hotspots.

The framework supports reproducible parameter sweeps over topology size, traffic load, number of QPs, and flowlet intervals, producing quantitative comparisons of routing efficiency and load balance.

---

## ⚙️ Installation

Clone and set up the environment:

    git clone git@github.com:KenziVisor/fat-tree-topology-sim-with-ecmp.git
    cd fat-tree-topology-sim-with-ecmp
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirements.txt

---

## Usage

### 1️⃣ Run Fat-Tree resilience experiment

    python3 fat-tree-topology-sim.py --experiment fat_tree -k 8 --sweep "0,0.05,0.1,0.2" --trials 10

This experiment evaluates network connectivity under random link failures.

---

### 2️⃣ Run classic ECMP experiment (Scenario A + B)

    python3 fat-tree-topology-sim.py --experiment ecmp -k 8 --num_flows 2000

This runs the static ECMP load-balancing experiment on two traffic scenarios.

---

### 3️⃣ Run RoCE routing experiment (ECMP vs e-ECMP vs Flowlet-e-ECMP)

Basic single-point experiment:

    python3 fat-tree-topology-sim.py --experiment roce -k 8 --num_flows 2000 --qps 8 --trials 20

Sweep over QPs (effect of e-ECMP):

    python3 fat-tree-topology-sim.py --experiment roce -k 8 --num_flows 2000 --qps_sweep 1,2,4,8,16 --trials 20

Sweep over flowlets (effect of flowlet reshuffling):

    python3 fat-tree-topology-sim.py --experiment roce -k 8 --num_flows 2000 --qps 8 --trials_sweep 1,2,5,10,20,40

Sweep over topology size:

    python3 fat-tree-topology-sim.py --experiment roce --k_sweep 4,8,16 --num_flows 2000 --qps 8 --trials 20

---

## Command-Line Arguments

| Argument | Meaning |
|----------|---------|
| `--experiment` | Select experiment: `fat_tree`, `ecmp`, or `roce` |
| `-k` | Fat-tree size (required unless using `--k_sweep` in RoCE) |
| `--k_sweep` | Comma-separated k values for RoCE sweep |
| `--num_flows` | Number of simulated flows (default: 2000) |
| `--flow_load` | Load per flow (default: 1.0) |
| `--qps` | Number of QPs per flow (for e-ECMP) |
| `--qps_sweep` | Comma-separated QP values for RoCE sweep |
| `--trials` | Number of flowlet time epochs |
| `--trials_sweep` | Comma-separated trial values for RoCE sweep |
| `--prob` | Link failure probability (fat_tree only) |
| `--sweep` | Probability sweep for fat_tree experiment |
| `--seed` | Random seed |

---

## Experiment 1: Link Failure Resilience

### What the Program Does

A **k-ary Fat-Tree** is constructed with four layers:

- Core layer  
- Aggregation layer  
- Edge layer  
- Host layer  

The program can:

- Draw the full topology
- Generate random host-to-host communication pairs
- Apply **random switch-to-switch link failures**
- Measure:
  - **Average shortest-path length**
  - **Reachable host-pair fraction**

---

### Interpreting the Graphs

Two curves are plotted:

- **Blue solid line** — Average shortest-path length  
- **Red dashed line** — Reachable host-pair fraction  

---

### Observations

The link-failure experiment reveals several important structural properties of the Fat-Tree topology:

- **At zero failure probability (`p = 0`)**, the network is fully connected and all host pairs communicate over minimum-length shortest paths.  
  This reflects the ideal non-blocking behavior of the Fat-Tree design.

- **As the link failure probability increases**, the network does not collapse immediately. Instead:
  - Many host pairs remain connected through alternative equal-cost paths.
  - Average shortest-path length increases gradually, indicating that traffic is being rerouted through longer but still valid paths.
  - This behavior demonstrates the effectiveness of the built-in path redundancy.

- **Connectivity degradation is gradual, not abrupt**:
  - Even at moderate failure probabilities, a large fraction of host pairs remain reachable.
  - Disconnections occur only after a significant portion of inter-switch links fail.
  - This shows that failures must accumulate before they significantly impact global connectivity.

- **Larger values of `k` significantly improve resilience**:
  - Increasing `k` increases the number of aggregation and core switches, and therefore the number of disjoint paths between hosts.
  - For the same failure probability, larger Fat-Trees maintain higher connectivity and shorter paths compared to smaller ones.
  - This confirms that scalability directly translates into fault tolerance.

Overall, the observed trends show that the Fat-Tree topology absorbs random failures by redistributing traffic across redundant paths, rather than concentrating failures into isolated bottlenecks.

---

### Conclusion

The link-failure experiment demonstrates that **Fat-Tree networks degrade gracefully under random switch-to-switch link failures**.

Instead of catastrophic disconnection, the topology exhibits:
- Progressive path elongation,
- Gradual loss of connectivity,
- Strong dependence on structural redundancy.

These properties explain why Fat-Tree architectures are widely adopted in real data centers:  
they provide not only high bandwidth under normal operation, but also **robust connectivity under failure**, allowing routing protocols to continue functioning effectively even when parts of the network are impaired.

---

## Output Files

| File | Description |
|------|-------------|
| `fat tree topology k = X p = 0.png` | Visualization of the network structure |
| `average lengths as a function of link failure probability k = X trials = Y.png` | Performance graph |

---

## Experiment 2: ECMP Load-Balancing Behavior

### Motivation

Even in a **non-blocking topology**, routing decisions can cause congestion.  
This experiment demonstrates how **static ECMP hashing alone** can create **severe load imbalance**, despite the availability of many equal-cost paths.

---

### ECMP Model

- Traffic is modeled as **unit-weight flows**
- Each flow is routed on **one shortest path**
- ECMP selects the path using a **deterministic hash of (src, dst)**
- Link load = **number of flows traversing the link**

This isolates the effect of **hash collisions**, without confounding traffic volume.

---

### Traffic Scenarios

#### Scenario A — Uniform Traffic

- Random source and destination hosts
- High entropy in ECMP hashing
- Expected behavior:
  - Even load distribution
  - Low congestion

#### Scenario B — Pod-to-Pod Traffic

- All flows originate in one pod and terminate in another
- Limited ECMP path diversity
- Expected behavior:
  - Hash collisions
  - Hot links and underutilized links elsewhere

---

### ECMP Visualization

The ECMP experiment produces a **layered Fat-Tree visualization** where:

- **Blue edges** → low load  
- **Red edges** → high load  
- Color scale is continuous

Each figure includes:
- ECMP load colorbar
- Scenario identifier
- Same layout as the original topology drawing

---

### Key Statistics

The following metrics are computed and used for analysis:

- **Maximum link load**
- **Average link load**
- **Standard deviation**
- **Max / Average ratio**
- **Fraction of overloaded links**

---

## ECMP Failure Analysis — Scenario B (Pod-to-Pod Traffic)

The figure "ecmp load visualization k=4 scenario=B" above visualizes the ECMP load distribution for **Scenario B**, where traffic is concentrated between two pods.  
Although the Fat-Tree topology provides multiple equal-cost paths between the pods, the resulting load distribution is **highly unbalanced**.

### Observed Behavior

A clear asymmetry can be observed at the **core layer**:

- Links connected to **`core_1`** are heavily congested (colored darked red).
- Links connected to **`core_3`** remain significantly less loaded (colored lighted orange).
- Aggregation and edge layers inherit this imbalance, creating end-to-end hot paths.

This imbalance occurs **despite the fact that the topology is non-blocking** and offers sufficient path diversity to evenly distribute the traffic.

### Root Cause: Static ECMP Hashing

The root cause of this behavior is the **static nature of ECMP**:

1. **Hash-based path selection**  
   ECMP assigns each flow to a single path using a deterministic hash over flow identifiers (e.g., source and destination).  
   It does **not** attempt to split traffic evenly across all available paths.

2. **No global load awareness**  
   ECMP makes routing decisions independently for each flow, without considering:
   - Current link utilization
   - Load on parallel paths
   - Whether traffic is already skewed toward a particular core switch

3. **No consideration of flow weights**  
   All flows are treated identically during hashing.  
   As a result:
   - Two large (elephant) flows can easily be hashed to the **same core switch and same path**
   - Instead of being evenly divided across available ECMP paths

In this scenario, many flows are hashed toward paths traversing **`core_1`**, while fewer flows are hashed toward **`core_3`**.  
Once this happens, ECMP **does not correct the imbalance**, even though alternative paths are underutilized.

### Key Insight

This experiment demonstrates that:

> **Even in a non-blocking Fat-Tree topology, static ECMP can create severe congestion due to hash collisions and lack of load awareness.**

The imbalance observed in the figure is not caused by insufficient capacity, but purely by **routing decisions**.  
This highlights a fundamental limitation of ECMP and motivates more advanced load-aware or flow-splitting routing mechanisms in modern data centers.

---

### Conclusion

This experiment shows that:

> **Static ECMP can create congestion even in a non-blocking Fat-Tree topology.**

The imbalance is caused purely by **hash collisions**, not by insufficient network capacity.

---

## Output Files

| File | Description |
|-----|-------------|
| `fat tree topology k equals X p equals 0.png` | Layered topology visualization |
| `ecmp load visualization k=X scenario=A.png` | ECMP load visualization (uniform traffic) |
| `ecmp load visualization k=X scenario=B.png` | ECMP load visualization (pod-to-pod traffic) |

---
## Experiment 3: RoCE Routing – ECMP, e-ECMP, and Flowlet-e-ECMP

### Motivation

Modern AI training clusters employ RDMA over Ethernet and rely on multi-path routing. 
Recent systems extend ECMP by splitting flows across multiple Queue Pairs (e-ECMP) and further reshuffling paths over time using flowlets. 
This experiment isolates and evaluates the **routing-level behavior** of these mechanisms.

---

### Routing Models

**ECMP:**  
Each flow is hashed once to a single equal-cost path.

**e-ECMP:**  
Each flow is split into multiple QPs, each hashed independently.

**Flowlet-based e-ECMP:**  
Each QP is periodically re-hashed across equal-cost paths at discrete flowlet intervals, enabling temporal path diversity while preserving in-order delivery within each flowlet.

---

### Traffic Model

- Uniform random host-to-host flows
- Equal normalized load per flow
- No queuing or transport modeling — routing effects only

---

### Evaluation Metrics

From per-link simulated loads:

- **p99 link load** – congestion hotspot severity  
- **CV (std/mean)** – global load balance fairness

---

### Output Visualization

For each parameter sweep (topology size k, QPs per flow, or number of flowlets), two graphs are produced:

- **p99(link load) vs sweep parameter**  
- **CV(load) vs sweep parameter**

Each graph compares three routing schemes:

**ECMP**, **e-ECMP**, and **Flowlet-e-ECMP**

---

### Output Files

Example filenames:

- **roce_p99_vs_qps_k=8_trials=20.png**
- **roce_cv_vs_qps_k=8_trials=20.png**
- **roce_p99_vs_trials_k=8_qps=8.png**
- **roce_cv_vs_k_qps=8_trials=20.png**

These plots are directly used for analysis in the final report.

---

## Summary

This simulator provides a lightweight experimental framework for evaluating multipath routing strategies in fat-tree data center networks. 
By comparing ECMP, e-ECMP, and flowlet-based e-ECMP under controlled traffic conditions, the project highlights the impact of increased path diversity and temporal reshuffling on load distribution.

The resulting measurements and visualizations offer clear insight into how modern routing extensions reduce congestion hotspots and improve global load balance, complementing transport-level studies of data center networks.

