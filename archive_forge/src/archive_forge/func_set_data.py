from collections import defaultdict
import networkx as nx
def set_data(self, data):
    """Inserts edges according to given sorted neighbor list.

        The input format is the same as the output format of get_data().

        Parameters
        ----------
        data : dict
            A dict mapping all nodes to a list of neighbors sorted in
            clockwise order.

        See Also
        --------
        get_data

        """
    for v in data:
        for w in reversed(data[v]):
            self.add_half_edge_first(v, w)