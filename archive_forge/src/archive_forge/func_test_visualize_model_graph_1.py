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
@unittest.skipUnless(matplotlib_available, 'matplotlib is not available.')
def test_visualize_model_graph_1(self):
    model = decode_model_1()
    community_map_object = detect_communities(model)
    with TempfileManager:
        fig, pos = community_map_object.visualize_model_graph(filename=TempfileManager.create_tempfile('test_visualize_model_graph_1.png'))
    correct_pos_dict_length = 5
    self.assertTrue(isinstance(pos, dict))
    self.assertEqual(len(pos), correct_pos_dict_length)