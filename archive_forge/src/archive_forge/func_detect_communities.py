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
def detect_communities(model, type_of_community_map='constraint', with_objective=True, weighted_graph=True, random_seed=None, use_only_active_components=True):
    """
    Detects communities in a Pyomo optimization model

    This function takes in a Pyomo optimization model and organizes the variables and constraints into a graph of nodes
    and edges. Then, by using Louvain community detection on the graph, a dictionary (community_map) is created, which
    maps (arbitrary) community keys to the detected communities within the model.

    Parameters
    ----------
    model: Block
        a Pyomo model or block to be used for community detection
    type_of_community_map: str, optional
        a string that specifies the type of community map to be returned, the default is 'constraint'.
        'constraint' returns a dictionary (community_map) with communities based on constraint nodes,
        'variable' returns a dictionary (community_map) with communities based on variable nodes,
        'bipartite' returns a dictionary (community_map) with communities based on a bipartite graph (both constraint
        and variable nodes)
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function is
        included in the model graph (and thus in 'community_map'); the default is True
    weighted_graph: bool, optional
        a Boolean argument that specifies whether community_map is created based on a weighted model graph or an
        unweighted model graph; the default is True (type_of_community_map='bipartite' creates an unweighted
        model graph regardless of this parameter)
    random_seed: int, optional
        an integer that is used as the random seed for the (heuristic) Louvain community detection
    use_only_active_components: bool, optional
        a Boolean argument that specifies whether inactive constraints/objectives are included in the community map

    Returns
    -------
    CommunityMap object (dict-like object)
        The CommunityMap object acts as a Python dictionary, mapping integer keys to tuples containing two lists
        (which contain the components in the given community) - a constraint list and variable list. Furthermore,
        the CommunityMap object stores relevant information about the given community map (dict), such as the model
        used to create it, its networkX representation, etc.
    """
    if not isinstance(model, ConcreteModel):
        raise TypeError("Invalid model: 'model=%s' - model must be an instance of ConcreteModel" % model)
    if type_of_community_map not in ('bipartite', 'constraint', 'variable'):
        raise TypeError("Invalid value for type_of_community_map: 'type_of_community_map=%s' - Valid values: 'bipartite', 'constraint', 'variable'" % type_of_community_map)
    if type(with_objective) != bool:
        raise TypeError("Invalid value for with_objective: 'with_objective=%s' - with_objective must be a Boolean" % with_objective)
    if type(weighted_graph) != bool:
        raise TypeError("Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph must be a Boolean" % weighted_graph)
    if random_seed is not None:
        if type(random_seed) != int:
            raise TypeError("Invalid value for random_seed: 'random_seed=%s' - random_seed must be a non-negative integer" % random_seed)
        if random_seed < 0:
            raise ValueError("Invalid value for random_seed: 'random_seed=%s' - random_seed must be a non-negative integer" % random_seed)
    if use_only_active_components is not True and use_only_active_components is not None:
        raise TypeError("Invalid value for use_only_active_components: 'use_only_active_components=%s' - use_only_active_components must be True or None" % use_only_active_components)
    model_graph, number_component_map, constraint_variable_map = generate_model_graph(model, type_of_graph=type_of_community_map, with_objective=with_objective, weighted_graph=weighted_graph, use_only_active_components=use_only_active_components)
    partition_of_graph = community_louvain.best_partition(model_graph, random_state=random_seed)
    number_of_communities = len(set(partition_of_graph.values()))
    community_map = {nth_community: [] for nth_community in range(number_of_communities)}
    for node in partition_of_graph:
        nth_community = partition_of_graph[node]
        community_map[nth_community].append(node)
    if type_of_community_map == 'bipartite':
        for community_key in community_map:
            constraint_node_list, variable_node_list = ([], [])
            node_community_list = community_map[community_key]
            for numbered_node in node_community_list:
                if numbered_node in constraint_variable_map:
                    constraint_node_list.append(number_component_map[numbered_node])
                else:
                    variable_node_list.append(number_component_map[numbered_node])
            community_map[community_key] = (constraint_node_list, variable_node_list)
    elif type_of_community_map == 'constraint':
        for community_key in community_map:
            constraint_list = sorted(community_map[community_key])
            variable_list = [constraint_variable_map[numbered_constraint] for numbered_constraint in constraint_list]
            variable_list = sorted(set([node for variable_sublist in variable_list for node in variable_sublist]))
            variable_list = [number_component_map[variable] for variable in variable_list]
            constraint_list = [number_component_map[constraint] for constraint in constraint_list]
            community_map[community_key] = (constraint_list, variable_list)
    elif type_of_community_map == 'variable':
        for community_key in community_map:
            variable_list = sorted(community_map[community_key])
            constraint_list = []
            for numbered_variable in variable_list:
                constraint_list.extend([constraint_key for constraint_key in constraint_variable_map if numbered_variable in constraint_variable_map[constraint_key]])
            constraint_list = sorted(set(constraint_list))
            constraint_list = [number_component_map[constraint] for constraint in constraint_list]
            variable_list = [number_component_map[variable] for variable in variable_list]
            community_map[community_key] = (constraint_list, variable_list)
    logger.info('%s communities were found in the model' % number_of_communities)
    if number_of_communities == 0:
        logger.error('in detect_communities: Empty community map was returned')
    if number_of_communities == 1:
        logger.warning('Community detection found that with the given parameters, the model could not be decomposed - only one community was found')
    return CommunityMap(community_map, type_of_community_map, with_objective, weighted_graph, random_seed, use_only_active_components, model, model_graph, number_component_map, constraint_variable_map, partition_of_graph)