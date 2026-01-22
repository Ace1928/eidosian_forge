
import networkx as nx

# Initialize Directed Graph for a mini graph neural network with 6 nodes plus an integrator
G = nx.DiGraph()

# Add nodes to the Graph representing a mini graph network with 6 nodes
mini_graph_nodes = [f'Node{i}' for i in range(1, 7)]  # Nodes 1 to 6 for the mini graph
G.add_nodes_from(mini_graph_nodes)

# Add directed edges between nodes in the mini graph
for node in mini_graph_nodes:
    for target in mini_graph_nodes:
        if node != target:
            G.add_edge(node, target)

# Add integrator node to the graph
G.add_node('Integrator')

# Add edges from each node in the mini graph to the integrator node
for node in mini_graph_nodes:
    G.add_edge(node, 'Integrator')

# Define a simple integrator function to aggregate inputs from mini graph nodes
def integrator_function(inputs):
    # Aggregating inputs (assuming binary signals from each node)
    aggregated_value = sum(inputs)
    # Implement a binary decision: fire (1) if aggregated value exceeds a threshold, else no fire (0)
    threshold = 3  # Example threshold
    return 1 if aggregated_value > threshold else 0

# Example inputs for the mini graph nodes, representing binary signals (0 or 1)
mini_graph_inputs = [1, 0, 1, 1, 0, 1]  # Example inputs for nodes
integrator_output = integrator_function(mini_graph_inputs)

print(f"The output of the integrator node is: {integrator_output}")  # Displaying the output of the integrator
