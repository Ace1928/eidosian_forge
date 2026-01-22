import itertools
import pytest
import networkx as nx
def test_strategy_saturation_largest_first(self):

    def color_remaining_nodes(G, colored_nodes, full_color_assignment=None, nodes_to_add_between_calls=1):
        color_assignments = []
        aux_colored_nodes = colored_nodes.copy()
        node_iterator = nx.algorithms.coloring.greedy_coloring.strategy_saturation_largest_first(G, aux_colored_nodes)
        for u in node_iterator:
            neighbour_colors = {aux_colored_nodes[v] for v in G[u] if v in aux_colored_nodes}
            for color in itertools.count():
                if color not in neighbour_colors:
                    break
            aux_colored_nodes[u] = color
            color_assignments.append((u, color))
            for i in range(nodes_to_add_between_calls - 1):
                if not len(color_assignments) + len(colored_nodes) >= len(full_color_assignment):
                    full_color_assignment_node, color = full_color_assignment[len(color_assignments) + len(colored_nodes)]
                    aux_colored_nodes[full_color_assignment_node] = color
                    color_assignments.append((full_color_assignment_node, color))
        return (color_assignments, aux_colored_nodes)
    for G, _, _ in SPECIAL_TEST_CASES['saturation_largest_first']:
        G = G()
        for nodes_to_add_between_calls in range(1, 5):
            colored_nodes = {}
            full_color_assignment, full_colored_nodes = color_remaining_nodes(G, colored_nodes)
            for ind, (node, color) in enumerate(full_color_assignment):
                colored_nodes[node] = color
                partial_color_assignment, partial_colored_nodes = color_remaining_nodes(G, colored_nodes, full_color_assignment=full_color_assignment, nodes_to_add_between_calls=nodes_to_add_between_calls)
                assert full_color_assignment[ind + 1:] == partial_color_assignment
                assert full_colored_nodes == partial_colored_nodes