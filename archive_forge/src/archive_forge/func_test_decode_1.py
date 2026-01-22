import logging
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import networkx_available, matplotlib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.contrib.community_detection.detection import (
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QP_simple import QP_simple
from pyomo.solvers.tests.models.LP_inactive_index import LP_inactive_index
from pyomo.solvers.tests.models.SOS1_simple import SOS1_simple
def test_decode_1(self):
    model = m = decode_model_1()
    test_community_maps, test_partitions = _collect_community_maps(model)
    correct_partitions = [{4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}]
    if correct_partitions == test_partitions:
        str_test_community_maps = [str(community_map) for community_map in test_community_maps]
        correct_community_maps = ["{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}"]
        self.assertEqual(correct_community_maps, str_test_community_maps)
    correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
    self.assertEqual(correct_num_communities, test_num_communities)
    self.assertEqual(correct_num_nodes, test_num_nodes)