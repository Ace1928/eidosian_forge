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
def test_communities_7(self):
    model = m = disconnected_model()
    test_community_maps, test_partitions = _collect_community_maps(model)
    correct_partitions = [{2: 0}, {2: 0, 3: 1, 4: 1}, {2: 0}, {2: 0, 3: 1, 4: 1}, {2: 0}, {2: 0, 3: 1, 4: 1}, {2: 0}, {2: 0, 3: 1, 4: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}]
    if correct_partitions == test_partitions:
        str_test_community_maps = [str(community_map) for community_map in test_community_maps]
        correct_community_maps = ["{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}", "{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}", "{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}", "{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}"]
        self.assertEqual(correct_community_maps, str_test_community_maps)
    correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
    self.assertEqual(correct_num_communities, test_num_communities)
    self.assertEqual(correct_num_nodes, test_num_nodes)