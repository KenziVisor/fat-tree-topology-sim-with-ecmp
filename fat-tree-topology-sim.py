import networkx as nx
import numpy as np
import argparse
import random
import sys
import os
from matplotlib import pyplot as plt


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


def get_hosts(g: nx.Graph) -> list[str]:
    """
    Return a sorted list of host nodes.
    """
    return sorted([n for n, d in g.nodes(data=True) if d.get("type") == "host"])


def sample_random_host_pairs(g: nx.Graph) -> list[tuple[str, str]]:
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


def compute_avg_shortest_path(g: nx.Graph, pairs: list[tuple[str, str]]):
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


def run_random_test_and_plot(k: int, prob_values: list[float], trials: int = 10):
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


