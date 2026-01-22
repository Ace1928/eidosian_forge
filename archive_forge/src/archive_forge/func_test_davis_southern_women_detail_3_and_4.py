import pytest
import networkx as nx
from networkx.algorithms.connectivity.kcomponents import (
def test_davis_southern_women_detail_3_and_4():
    solution = {3: [{'Nora Fayette', 'E10', 'Myra Liddel', 'E12', 'E14', 'Frances Anderson', 'Evelyn Jefferson', 'Ruth DeSand', 'Helen Lloyd', 'Eleanor Nye', 'E9', 'E8', 'E5', 'E4', 'E7', 'E6', 'E1', 'Verne Sanderson', 'E3', 'E2', 'Theresa Anderson', 'Pearl Oglethorpe', 'Katherina Rogers', 'Brenda Rogers', 'E13', 'Charlotte McDowd', 'Sylvia Avondale', 'Laura Mandeville'}], 4: [{'Nora Fayette', 'E10', 'Verne Sanderson', 'E12', 'Frances Anderson', 'Evelyn Jefferson', 'Ruth DeSand', 'Helen Lloyd', 'Eleanor Nye', 'E9', 'E8', 'E5', 'E4', 'E7', 'E6', 'Myra Liddel', 'E3', 'Theresa Anderson', 'Katherina Rogers', 'Brenda Rogers', 'Charlotte McDowd', 'Sylvia Avondale', 'Laura Mandeville'}]}
    G = nx.davis_southern_women_graph()
    result = nx.k_components(G)
    for k, components in result.items():
        if k < 3:
            continue
        assert len(components) == len(solution[k])
        for component in components:
            assert component in solution[k]