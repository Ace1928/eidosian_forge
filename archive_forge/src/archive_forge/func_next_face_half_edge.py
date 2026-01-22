from collections import defaultdict
import networkx as nx
def next_face_half_edge(self, v, w):
    """Returns the following half-edge left of a face.

        Parameters
        ----------
        v : node
        w : node

        Returns
        -------
        half-edge : tuple
        """
    new_node = self[w][v]['ccw']
    return (w, new_node)