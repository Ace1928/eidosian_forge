import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def orthogonal_draw(self, viewer=None, show_crossing_labels=False):
    """
    Opens a Plink link viewer window displaying the current link.
    The strands of the links are unions of edges in the standard
    integer grid, following the work of `Tamassia
    <https://dx.doi.org/10.1137/0216030>`_ and `Bridgeman
    et. al. <ftp://ftp.cs.brown.edu/pub/techreports/99/cs99-04.pdf>`_
    """
    if not plink:
        print('To view links install PLink version 2.0.2 or higher.')
        return
    if viewer is None:
        viewer = plink.LinkDisplay(show_crossing_labels=show_crossing_labels)
    diagram = OrthogonalLinkDiagram(self)
    viewer.unpickle(*diagram.plink_data())
    try:
        viewer.zoom_to_fit()
    except AttributeError:
        pass
    return viewer