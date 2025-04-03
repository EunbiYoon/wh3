from graphviz import Digraph
import json
import os

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

def count_all_files(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files

# 🌀 모든 json 파일을 하나씩 input_file로 넣어 시각화
base_dir = "ntrees_50"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".json") and file.startswith("tree_"):
            input_file = os.path.join(root, file)
            output_file = os.path.splitext(input_file)[0]  # 확장자 제거

            try:
                with open(input_file, "r") as f:
                    tree_data = json.load(f)
                visualize_tree_from_dict(tree_data, output_file)
            except Exception as e:
                print(f"❌ Error with file {input_file}: {e}")