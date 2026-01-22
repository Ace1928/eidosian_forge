from networkx.utils.misc import graphs_equal
import pytest
import networkx as nx
import cirq
def test_distance_on_device_and_is_adjacent():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    q_int = [mm.logical_qid_to_int[q[i]] if q[i] in initial_mapping else -1 for i in range(len(q))]
    assert mm.dist_on_device(q_int[1], q_int[3]) == 1
    assert mm.is_adjacent(q_int[1], q_int[3])
    assert mm.dist_on_device(q_int[1], q_int[2]) == 2
    assert mm.is_adjacent(q_int[1], q_int[2]) is False
    assert mm.dist_on_device(q_int[1], q_int[4]) == 3
    mm.apply_swap(q_int[2], q_int[3])
    assert mm.dist_on_device(q_int[1], q_int[3]) == 2
    assert mm.is_adjacent(q_int[1], q_int[3]) is False
    assert mm.dist_on_device(q_int[1], q_int[2]) == 1
    assert mm.is_adjacent(q_int[1], q_int[2])
    assert mm.dist_on_device(q_int[1], q_int[4]) == 3