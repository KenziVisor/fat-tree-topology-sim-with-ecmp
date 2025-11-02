import networkx as nx
import argparse
import sys
import os
from matplotlib import pyplot as plt
from numba.core.ir import Raise


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


def draw_graph(g: nx.Graph, k:int) -> None:
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
    file_name = f'fat tree topology k equals {k}.png'
    plt.savefig(file_name)
    if os.path.exists(file_name):
        size = os.path.getsize(file_name)
        if size > 0:
            print(f"✅ File '{file_name}' exists and has size {size} bytes.")
            return
    exit(-1)
    

if __name__ == "__main__":
    #I used chatGPT for the parser of parameter k

    # --- Parse arguments ---
    parser = argparse.ArgumentParser(
        description="Generate and draw a k-ary Fat-Tree topology."
    )
    parser.add_argument(
        "-k",
        type=int,
        required=True,
        help="Fat-tree parameter k (must be an even positive integer)."
    )
    args = parser.parse_args()

    # --- Validate k ---
    k = args.k
    if k <= 0 or k % 2 != 0:
        print("Error: parameter k must be a positive even integer.", file=sys.stderr)
        sys.exit(1)

    g = build_fat_tree(k)
    draw_graph(g, k)
