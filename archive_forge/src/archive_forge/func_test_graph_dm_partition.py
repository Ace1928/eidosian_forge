import pyomo.common.unittest as unittest
from pyomo.common.dependencies import networkx as nx, networkx_available
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
def test_graph_dm_partition(self):
    graph, top_nodes = self._construct_graph()
    with self.assertRaises(nx.AmbiguousSolution):
        top_dmp, bot_dmp = dulmage_mendelsohn(graph)
    top_dmp, bot_dmp = dulmage_mendelsohn(graph, top_nodes=top_nodes)
    self.assertFalse(nx.is_connected(graph))
    underconstrained_top = {0, 1}
    underconstrained_bot = {7, 8, 9}
    self.assertEqual(underconstrained_top, set(top_dmp[2]))
    self.assertEqual(underconstrained_bot, set(bot_dmp[0] + bot_dmp[1]))
    overconstrained_top = {5, 6}
    overconstrained_bot = {13}
    self.assertEqual(overconstrained_top, set(top_dmp[0] + top_dmp[1]))
    self.assertEqual(overconstrained_bot, set(bot_dmp[2]))
    wellconstrained_top = {2, 3, 4}
    wellconstrained_bot = {10, 11, 12}
    self.assertEqual(wellconstrained_top, set(top_dmp[3]))
    self.assertEqual(wellconstrained_bot, set(bot_dmp[3]))