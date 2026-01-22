from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def source_dest_peer(self, arc, name, index=None):
    """
        Return the object that is the peer to the source port's member.
        This is either the destination port's member, or the variable
        on the arc's expanded block for Extensive properties. This will
        return the appropriate index of the peer.
        """
    if arc.src.is_extensive(name):
        evar = arc.expanded_block.component(name)
        if evar is not None:
            return evar[index]
    mem = arc.dest.vars[name]
    if mem.is_indexed():
        return mem[index]
    else:
        return mem