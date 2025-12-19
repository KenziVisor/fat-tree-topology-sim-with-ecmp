import networkx as nx
import numpy as np
import argparse
import random
import sys
import os
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from typing import List, Tuple


def build_fat_tree(k: int) -> nx.Graph:
    """
    :param k: The parameter k as mentioned in the paper.
    :return: return the k-ary fat tree topology as a networkx graph object
    """
    #Expanding the input parameter K
    pods_len = k
    aggregation_layer_len = k//2
    edge_layer_len = k//2
    core_switches_len = (k//2)**2
    hosts_per_edge_switch = k//2

    #Building the topology
    g = nx.Graph()

    #Create the core switches
    for i in range(core_switches_len):
        core_name = f"core_{i}"
        g.add_node(core_name, type="core")

    #create for each pod the aggregation switches, edge switches and hosts
    for pod in range(pods_len):
        for i in range(aggregation_layer_len):
            agg_name = f"pod_{pod}_agg_{i}"
            g.add_node(agg_name, type="agg")
            for j in range(i*k//2, (i+1)*k//2):
                g.add_edge(agg_name, f"core_{j}")
        for i in range(edge_layer_len):
            edge_name = f"pod_{pod}_edge_{i}"
            g.add_node(edge_name, type="edge")
            for j in range(aggregation_layer_len):
                g.add_edge(edge_name, f"pod_{pod}_agg_{j}")
            for j in range(hosts_per_edge_switch):
                host_name = f"pod_{pod}_edge_{i}_host_{j}"
                g.add_node(host_name, type="host")
                g.add_edge(edge_name, host_name)
    return g


def draw_graph(g: nx.Graph, k: int, p: float) -> None:
    """
    I used chatGPT in order to create this method:
    Draws the given fat-tree graph with clear layers:
    Core (top), Aggregation, Edge, Hosts (bottom).
    Nodes are aligned in horizontal lines per layer.
    """

    # Separate nodes by their type
    layers = {"core": [], "agg": [], "edge": [], "host": []}
    for n, d in g.nodes(data=True):
        node_type = d.get("type")
        if node_type in layers:
            layers[node_type].append(n)

    # Sort nodes within each layer for a cleaner layout
    for t in layers:
        layers[t] = sorted(layers[t])

    # Assign y-coordinates for each layer
    y_positions = {"core": 3, "agg": 2, "edge": 1, "host": 0}

    # Create position dict
    pos = {}
    for layer_name, nodes in layers.items():
        count = len(nodes)
        if count == 0:
            continue
        # Spread nodes evenly along x-axis
        for i, node in enumerate(nodes):
            pos[node] = (i - count / 2.0, y_positions[layer_name])

    # Color by type
    color_map = []
    for n in g.nodes():
        t = g.nodes[n]["type"]
        if t == "core":
            color_map.append("skyblue")
        elif t == "agg":
            color_map.append("lightgreen")
        elif t == "edge":
            color_map.append("orange")
        elif t == "host":
            color_map.append("lightgray")
        else:
            color_map.append("white")

    # Draw
    plt.figure(figsize=(12, 7), constrained_layout=True)
    nx.draw(
        g,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=700,
        edgecolors="black",
        font_size=8,
        font_weight="bold",
        width=0.8,
        alpha=0.9,
    )

    # Titles and aesthetics
    plt.title("Fat-Tree Topology — Layered View (Core → Agg → Edge → Hosts)")
    plt.axis("off")
    file_name = f'fat tree topology k equals {k} p equals {p}.png'
    plt.savefig(file_name)
    if os.path.exists(file_name):
        size = os.path.getsize(file_name)
        if size > 0:
            print(f"✅ File '{file_name}' exists and has size {size} bytes.")
            return
    exit(1)
    

def apply_failures(g: nx.Graph, prob: float) -> nx.Graph:
    '''
    :param g: The original graph
    :param prob: Probability of link failure
    :return: Return the new graph h with link failure
    '''
    if prob < 0 or prob > 1:
        raise ValueError("p must be in [0, 1] (probability of link failure).")
    h = g.copy()
    to_remove = []
    for u, v in h.edges():
        type_u = h.nodes[u].get("type")
        type_v = h.nodes[v].get("type")
        # Only remove if both endpoints are switches (no host links)
        if type_u != "host" and type_v != "host":
            if random.random() < prob:
                to_remove.append((u, v))
    h.remove_edges_from(to_remove)
    return h


def get_hosts(g: nx.Graph) -> List[str]:
    """
    Return a sorted list of host nodes.
    """
    return sorted([n for n, d in g.nodes(data=True) if d.get("type") == "host"])


def sample_random_host_pairs(g: nx.Graph) -> List[Tuple[str, str]]:
    hosts = get_hosts(g)
    n = len(hosts)

    if n % 2 != 0:
        raise ValueError("Number of hosts must be even to form pairs.")

    # Step 3: make shuffled index list
    idxs = list(range(n))
    random.shuffle(idxs)

    # Step 4: pair them
    pairs = []
    for i in range(0, n, 2):
        a = idxs[i]
        b = idxs[i + 1]
        pairs.append((hosts[a], hosts[b]))

    return pairs


def compute_avg_shortest_path(g: nx.Graph, pairs: List[Tuple[str, str]]):
    """
    Compute average shortest path over all reachable pairs.
    """
    lengths = []
    for u, v in pairs:
        if nx.has_path(g, u, v):
            d = nx.shortest_path_length(g, u, v)
            lengths.append(d)

    total = len(pairs)
    reachable = len(lengths)
    connected_frac = reachable / total if total > 0 else 0.0
    avg_len = np.mean(lengths) if reachable > 0 else np.nan

    return avg_len, connected_frac


def run_random_test_and_plot(k: int, prob_values: List[float], trials: int = 10):
    base = build_fat_tree(k)

    avg_lengths = []
    conn_fracs = []

    for prob in prob_values:
        trial_lengths = []
        trial_connfracs = []

        for _ in range(trials):
            # 1-to-1 random pairs (fresh mapping per prob, as in Table 2 averaging behavior)
            pairs = sample_random_host_pairs(base)
            # apply failures
            h = apply_failures(base, prob)
            # metric
            mean_len, conn_frac = compute_avg_shortest_path(h, pairs)
            if not np.isnan(mean_len):
                trial_lengths.append(mean_len)
            trial_connfracs.append(conn_frac)

        avg_lengths.append(np.mean(trial_lengths) if trial_lengths else np.nan)
        conn_fracs.append(np.mean(trial_connfracs))

    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)

    # Left y-axis: average shortest path
    line1 = ax1.plot(prob_values, avg_lengths, marker="o", color="blue", label="Avg Shortest-Path Length")
    ax1.set_xlabel("Link Failure Probability (prob)")
    ax1.set_ylabel("Average Shortest-Path Length", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Right y-axis: reachable fraction
    ax2 = ax1.twinx()
    line2 = ax2.plot(prob_values, conn_fracs, marker="x", linestyle="--", color="red", label="Reachable Pair Fraction")
    ax2.set_ylabel("Reachable Fraction", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title(f"Random 1-to-1 — Avg Shortest Path & Reachability vs prob (k={k} and trials={trials})")

    file_name = f'average lengths as a function of link failure probability k {k} trials {trials}.png'
    plt.savefig(file_name)
    if os.path.exists(file_name):
        size = os.path.getsize(file_name)
        if size > 0:
            print(f"✅ File '{file_name}' exists and has size {size} bytes.")
            return
    exit(1)


def extract_topology_info(g: nx.Graph):
    print("[ECMP] Extracting topology info...")

    hosts = sorted([n for n, d in g.nodes(data=True) if d.get("type") == "host"])
    if not hosts:
        raise ValueError("ECMP experiment: No hosts found in the graph")

    core = [n for n, d in g.nodes(data=True) if d.get("type") == "core"]
    agg = [n for n, d in g.nodes(data=True) if d.get("type") == "agg"]
    edge = [n for n, d in g.nodes(data=True) if d.get("type") == "edge"]

    print(f"Hosts={len(hosts)}, Edge={len(edge)}, Agg={len(agg)}, Core={len(core)}")

    state = {
        "hosts": hosts,
        "core": core,
        "agg": agg,
        "edge": edge,
    }

    print("[ECMP] Topology info ready.\n")
    return state


def generate_flows(state, num_flows, scenario="A"):
    """
    Generate a list of (src, dst) host pairs.
    Scenario A: uniform all-to-all random traffic.
    Scenario B: concentrated pod-to-pod traffic.
    """

    hosts = state["hosts"]

    # -------------------------------
    # Scenario A — uniform random
    # -------------------------------
    if scenario.upper() == "A":
        flows = []
        for _ in range(num_flows):
            src, dst = random.sample(hosts, 2)
            flows.append((src, dst))
        return flows

    # -------------------------------
    # Scenario B — pod-to-pod traffic
    # -------------------------------
    elif scenario.upper() == "B":

        # Infer pod ID from the host name:
        # Assumes host names look like: "pod_3_edge_1_host_0"
        pod_map = {}
        for h in hosts:
            parts = h.split("_")
            if len(parts) < 2 or parts[0] != "pod":
                raise ValueError(f"Unexpected host name format: {h}")
            pod_id = int(parts[1])
            pod_map.setdefault(pod_id, []).append(h)

        if len(pod_map) < 2:
            raise ValueError("Need at least 2 pods for Scenario B traffic")

        # Pick two different pods
        pod_A, pod_B = random.sample(list(pod_map.keys()), 2)

        src_hosts = pod_map[pod_A]
        dst_hosts = pod_map[pod_B]

        flows = []
        for _ in range(num_flows):
            src = random.choice(src_hosts)
            dst = random.choice(dst_hosts)
            flows.append((src, dst))

        return flows

    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Use 'A' or 'B'.")


def compute_all_ecmp_paths(g, flows):
    """
    For each (src, dst), compute all equal-cost shortest paths.
    Returns: dict where key=(src, dst) and value=list_of_paths.
    Each path is a list of nodes.
    """
    print("[ECMP] Computing all ECMP shortest paths...")

    path_map = {}

    for (src, dst) in flows:
        try:
            # networkx.all_shortest_paths yields all equal-cost paths
            paths = list(nx.all_shortest_paths(g, source=src, target=dst))
        except nx.NetworkXNoPath:
            raise ValueError(f"No path found between {src} and {dst}")

        # store list of node-paths
        path_map[(src, dst)] = paths

    print(f"[ECMP] Finished computing ECMP paths for {len(flows)} flows.\n")
    return path_map


def hash_and_route(g, flows, path_map):
    print("[ECMP] Hashing flows and assigning ECMP paths...")

    # link load map: (u, v) → load_value
    loads = {}

    for (src, dst) in flows:
        paths = path_map[(src, dst)]
        num_paths = len(paths)

        # deterministic ECMP hash
        h = (hash(src) * 1315423911 + hash(dst) * 2654435761) % num_paths
        chosen_path = paths[h]

        # accumulate load on each directed link in the chosen path
        for i in range(len(chosen_path) - 1):
            u = chosen_path[i]
            v = chosen_path[i + 1]

            # store links as sorted tuples so (u, v) and (v, u) are treated consistently
            link = tuple(sorted((u, v)))

            loads[link] = loads.get(link, 0) + 1

    print("[ECMP] Flow assignment complete.\n")
    return loads


def compute_load_stats(loads):
    """
    Input: loads dict where key = (u, v), value = load_count.
    Returns a dict with basic utilization statistics.
    """
    if not loads:
        return {
            "max": 0,
            "avg": 0,
            "std": 0,
            "num_links": 0,
            "over_1": 0
        }

    arr = np.array(list(loads.values()), dtype=float)

    stats = {
        "max": float(arr.max()),
        "avg": float(arr.mean()),
        "std": float(arr.std()),   # population std (matches earlier behavior)
        "num_links": int(arr.size),
        "over_1": int(np.sum(arr > 1))
    }

    print("[ECMP] Load statistics:")
    print(stats, "\n")

    return stats


def draw_ecmp_graph(g: nx.Graph, k: int, loads: dict, scenario: str, stats: dict) -> None:
    """
    Draws the fat-tree graph in the exact same layered style as draw_graph(),
    but edges are colored according to ECMP load (blue→red).
    Saves the file and does NOT call plt.show().
    """

    # --- 1. Separate nodes by layer ---
    layers = {"core": [], "agg": [], "edge": [], "host": []}
    for n, d in g.nodes(data=True):
        t = d.get("type")
        if t in layers:
            layers[t].append(n)

    for t in layers:
        layers[t] = sorted(layers[t])

    # --- 2. Node positions ---
    y_positions = {"core": 3, "agg": 2, "edge": 1, "host": 0}
    pos = {}

    for layer_name, nodes in layers.items():
        count = len(nodes)
        if count == 0:
            continue
        for i, node in enumerate(nodes):
            pos[node] = (i - count / 2.0, y_positions[layer_name])

    # --- 3. Node colors ---
    color_map = []
    for n in g.nodes():
        t = g.nodes[n]["type"]
        if t == "core":
            color_map.append("skyblue")
        elif t == "agg":
            color_map.append("lightgreen")
        elif t == "edge":
            color_map.append("orange")
        elif t == "host":
            color_map.append("lightgray")
        else:
            color_map.append("white")

    # --- 4. Edge color spectrum based on ECMP load ---
    all_edges = list(g.edges())
    edge_loads = []
    for (u, v) in all_edges:
        key = tuple(sorted((u, v)))
        edge_loads.append(loads.get(key, 0))

    max_load = max(edge_loads) if edge_loads else 1
    if max_load == 0:
        max_load = 1

    cmap = plt.cm.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=0, vmax=max_load)
    edge_colors = [cmap(norm(v)) for v in edge_loads]

    # --- 5. Draw the graph ---
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    nx.draw(
        g,
        pos,
        ax=ax,
        with_labels=True,
        node_color=color_map,
        node_size=700,
        edge_color=edge_colors,
        edgecolors="black",
        font_size=8,
        font_weight="bold",
        width=1.2,
        alpha=0.9,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="ECMP Load")

    plt.title(f"ECMP Load Visualization — Scenario {scenario}, Avg load: {stats['avg']:.2f}, Max load: {stats['max']:.2f}, Std load: {stats['std']:.2f}")
    plt.axis("off")

    # --- 6. Save ---
    file_name = f"ecmp load visualization k={k} scenario={scenario}.png"
    plt.savefig(file_name)

    if os.path.exists(file_name):
        size = os.path.getsize(file_name)
        if size > 0:
            print(f"✅ File '{file_name}' saved ({size} bytes).")
            return
    exit(1)


def run_ecmp_experiment(g: nx.Graph, num_flows=2000, scenario="A"):
    print("\n=== ECMP EXPERIMENT START ===")

    state = extract_topology_info(g)
    flows = generate_flows(state, num_flows, scenario)
    path_map = compute_all_ecmp_paths(g, flows)
    loads = hash_and_route(g, flows, path_map)
    stats = compute_load_stats(loads)
    draw_ecmp_graph(g, k, loads, scenario, stats)

    print("=== ECMP EXPERIMENT END ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random test: Avg shortest path vs. link failure prob.")
    parser.add_argument("-k", "--k", type=int, required=True,
                        help="Fat-tree parameter k (positive even integer).")
    parser.add_argument("-p", "--prob", type=float, default=0.0,
                        help="Single failure probability in [0,1] (default: 0).")
    parser.add_argument("--sweep", type=str, default="",
                        help='Comma-separated probs to sweep, e.g. "0,0.01,0.02,0.05" (overrides -p).')
    parser.add_argument("-t", "--trials", type=int, default=10,
                        help="Number of averaging trials per probability value (default=10).")
    parser.add_argument("--ecmp", type=int, choices=[0, 1], default=0,
                        help="If 1, run ECMP experiment (only uses -k); if 0 or omitted, run original experiments")
    args = parser.parse_args()

    k = args.k
    if k <= 0 or k % 2 != 0:
        print("Error: k must be a positive even integer.", file=sys.stderr)
        sys.exit(1)

    trials = args.trials
    if trials < 1:
        print("Error: --trials must be an integer >= 1.", file=sys.stderr)
        sys.exit(1)

    g = build_fat_tree(k)
    draw_graph(g, k, p=0)

    if args.ecmp == 1:
        run_ecmp_experiment(g, num_flows=2000, scenario="A")
        run_ecmp_experiment(g, num_flows=2000, scenario="B")
        exit(0)

    if args.sweep:
        try:
            prob_values = [float(x) for x in args.sweep.split(",")]
        except ValueError:
            print("Error: --sweep must be comma-separated floats.", file=sys.stderr)
            sys.exit(1)
        if not all(0.0 <= x <= 1.0 for x in prob_values):
            print("Error: all sweep probabilities must be in [0,1].", file=sys.stderr)
            sys.exit(1)
        run_random_test_and_plot(k, prob_values, trials)
    else:
        prob = args.prob
        if not (0.0 <= prob <= 1.0):
            print("Error: prob must be in [0,1].", file=sys.stderr)
            sys.exit(1)
        run_random_test_and_plot(k, [prob], trials)


