import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def repair_components(self):
    """
        Repair the link components and store their numbering for
        easy reference.  Also order the strand_CEPs so they go component
        by component.
        """
    self.link.link_components = [component[0].component() for component in self.link.link_components]
    self.strand_CEP_to_component = stc = {}
    self.strand_CEPs = []
    for n, component in enumerate(self.link.link_components):
        for ce in component:
            if isinstance(ce.crossing, Strand):
                stc[ce] = n
                self.strand_CEPs.append(ce)