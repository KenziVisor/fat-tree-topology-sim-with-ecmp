# fat-tree-topology-sim

\# 🕸️ Fat-Tree Topology Generator and Visualizer



This project implements a \*\*k-ary fat-tree network topology\*\* using Python and `networkx`.  

It was built as part of \*Homework 1 — Fat-Tree Network Topology Generation and Analysis\*, based on the design proposed in  

\*\*Al-Fares et al., “A Scalable, Commodity Data Center Network Architecture,” SIGCOMM 2008.\*\*



---



\## 🧭 Overview

A \*\*k-ary fat-tree\*\* is a three-tier network architecture (Core → Aggregation → Edge → Hosts) that achieves full bisection bandwidth using only commodity switches.  

Each switch has \*k\* ports — half used for upward links and half for downward links.



This implementation:

\- Programmatically builds the full topology for a given even \*k\*  

\- Draws it in a clean, layered layout using `matplotlib`  



---



\## 🧮 Topology Structure Formulas



| Element | Formula | Description |

|----------|----------|-------------|

| Pods | `k` | Each pod contains `k/2` edge + `k/2` aggregation switches |

| Core switches | `(k/2)²` | Divided into `k/2` groups of `k/2` switches |

| Total switches | `(5 × k²) / 4` | All switches = pods × switches per pod + core |

| Hosts per edge | `k/2` | Each edge switch connects to `k/2` hosts |

| Total hosts | `k³ / 4` | Maximum end hosts supported |

| Parallel paths | `k/2` | Equal-cost paths between pods |



Example for `k = 4`:

\- 4 pods, 4 core switches  

\- 16 switches total, 16 hosts  



---



\## ⚙️ Installation


```bash

git clone git@github.com:KenziVisor/fat-tree-topology-sim.git
cd fat-tree-topology-sim
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install networkx matplotlib argparse
python3 fat-tree-topology-sim.py -k 4