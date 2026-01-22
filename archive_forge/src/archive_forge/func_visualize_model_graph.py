from logging import getLogger
from pyomo.common.dependencies import attempt_import
from pyomo.core import (
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.common.dependencies import networkx as nx
from pyomo.common.dependencies.matplotlib import pyplot as plt
from itertools import combinations
import copy
def visualize_model_graph(self, type_of_graph='constraint', filename=None, pos=None):
    """
        This function draws a graph of the communities for a Pyomo model.

        The type_of_graph parameter is used to create either a variable-node graph, constraint-node graph, or
        bipartite graph of the Pyomo model. Then, the nodes are colored based on the communities they are in - which
        is based on the community map (self.community_map). A filename can be provided to save the figure, otherwise
        the figure is illustrated with matplotlib.

        Parameters
        ----------
        type_of_graph: str, optional
            a string that specifies the types of nodes drawn on the model graph, the default is 'constraint'.
            'constraint' draws a graph with constraint nodes,
            'variable' draws a graph with variable nodes,
            'bipartite' draws a bipartite graph (with both constraint and variable nodes)
        filename: str, optional
            a string that specifies a path for the model graph illustration to be saved
        pos: dict, optional
            a dictionary that maps node keys to their positions on the illustration

        Returns
        -------
        fig: matplotlib figure
            the figure for the model graph drawing
        pos: dict
            a dictionary that maps node keys to their positions on the illustration - can be used to create consistent
            layouts for graphs of a given model
        """
    assert type_of_graph in ('bipartite', 'constraint', 'variable'), "Invalid graph type specified: 'type_of_graph=%s' - Valid values: 'bipartite', 'constraint', 'variable'" % type_of_graph
    assert isinstance(filename, (type(None), str)), "Invalid value for filename: 'filename=%s' - filename must be a string" % filename
    if type_of_graph != self.type_of_community_map:
        model_graph, number_component_map, constraint_variable_map = generate_model_graph(self.model, type_of_graph=type_of_graph, with_objective=self.with_objective, weighted_graph=self.weighted_graph, use_only_active_components=self.use_only_active_components)
    else:
        model_graph, number_component_map, constraint_variable_map = (self.graph, self.graph_node_mapping, self.constraint_variable_map)
    component_number_map = ComponentMap(((comp, number) for number, comp in number_component_map.items()))
    numbered_community_map = copy.deepcopy(self.community_map)
    for key in self.community_map:
        numbered_community_map[key] = ([component_number_map[component] for component in self.community_map[key][0]], [component_number_map[component] for component in self.community_map[key][1]])
    if type_of_graph == 'bipartite':
        list_of_node_lists = [list_of_nodes for list_tuple in numbered_community_map.values() for list_of_nodes in list_tuple]
        node_list = [node for sublist in list_of_node_lists for node in sublist]
        color_list = []
        for node in node_list:
            not_found = True
            for community_key in numbered_community_map:
                if not_found and node in numbered_community_map[community_key][0] + numbered_community_map[community_key][1]:
                    color_list.append(community_key)
                    not_found = False
        if model_graph.number_of_nodes() > 0 and nx.is_connected(model_graph):
            top_nodes = nx.bipartite.sets(model_graph)[1]
        else:
            top_nodes = {node for node in model_graph.nodes() if node in constraint_variable_map}
        if pos is None:
            pos = nx.bipartite_layout(model_graph, top_nodes)
    else:
        position = 0 if type_of_graph == 'constraint' else 1
        list_of_node_lists = list((i[position] for i in numbered_community_map.values()))
        node_list = [node for sublist in list_of_node_lists for node in sublist]
        color_list = []
        for node in node_list:
            not_found = True
            for community_key in numbered_community_map:
                if not_found and node in numbered_community_map[community_key][position]:
                    color_list.append(community_key)
                    not_found = False
        if pos is None:
            pos = nx.spring_layout(model_graph)
    color_map = plt.cm.get_cmap('viridis', len(numbered_community_map))
    fig = plt.figure()
    nx.draw_networkx_nodes(model_graph, pos, nodelist=node_list, node_size=40, cmap=color_map, node_color=color_list)
    nx.draw_networkx_edges(model_graph, pos, alpha=0.5)
    graph_type = type_of_graph.capitalize()
    community_map_type = self.type_of_community_map.capitalize()
    main_graph_title = '%s graph - colored using %s community map' % (graph_type, community_map_type)
    main_font_size = 14
    plt.suptitle(main_graph_title, fontsize=main_font_size)
    subtitle_naming_dict = {'bipartite': 'Nodes are variables and constraints & Edges are variables in a constraint', 'constraint': 'Nodes are constraints & Edges are common variables', 'variable': 'Nodes are variables & Edges are shared constraints'}
    subtitle_font_size = 11
    plt.title(subtitle_naming_dict[type_of_graph], fontsize=subtitle_font_size)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
    return (fig, pos)