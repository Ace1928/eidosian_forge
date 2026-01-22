import abc
from collections import namedtuple
from typing import Optional
from torch.fx.graph_module import GraphModule
from torch.fx._compatibility import compatibility
def ensures(self, graph_module: GraphModule) -> None:
    """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
    pass