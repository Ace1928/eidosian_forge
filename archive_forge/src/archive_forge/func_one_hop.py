import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def one_hop(self, Gx, Gx_node, core_x, pred, succ):
    """
        The ego node.
        """
    pred_dates = self.get_pred_dates(Gx, Gx_node, core_x, pred)
    succ_dates = self.get_succ_dates(Gx, Gx_node, core_x, succ)
    return self.test_one(pred_dates, succ_dates) and self.test_two(pred_dates, succ_dates)