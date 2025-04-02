from graphviz import Digraph
import json

def visualize_tree_from_dict(tree_dict, output_file="tree"):
    dot = Digraph()
    
    def add_nodes_edges(tree, parent=None, edge_label=""):
        node_id = str(id(tree))  # unique id for Graphviz

        if "label" in tree:
            dot.node(node_id, f"Label: {tree['label']}", shape="box", style="filled", color="lightblue")
        else:
            dot.node(node_id, f"Feature: {tree['feature']}", shape="ellipse", style="filled", color="lightgreen")

        if parent:
            dot.edge(parent, node_id, label=edge_label)

        if "children" in tree:
            for value, subtree in tree["children"].items():
                add_nodes_edges(subtree, parent=node_id, edge_label=str(value))

    add_nodes_edges(tree_dict)
    dot.render(output_file, format="png", cleanup=True)
    print(f"✅ Saved visualization to {output_file}.png")

# 사용 예시
input_file="ntrees_1/tree_1.json"
output_file="ntrees_1/ntrees_1.png"
with open(input_file, "r") as f:
    tree_data = json.load(f)

visualize_tree_from_dict(tree_data, output_file)