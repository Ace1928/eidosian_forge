import os
import os.path as op
import sys
from datetime import datetime
from copy import deepcopy
import pickle
import shutil
import numpy as np
from ... import config, logging
from ...utils.misc import str2bool
from ...utils.functions import getsource, create_function_from_source
from ...interfaces.base import traits, TraitedSpec, TraitDictObject, TraitListObject
from ...utils.filemanip import save_json
from .utils import (
from .base import EngineBase
from .nodes import MapNode
def remove_nodes(self, nodes):
    """Remove nodes from a workflow

        Parameters
        ----------
        nodes : list
            A list of EngineBase-based objects
        """
    self._graph.remove_nodes_from(nodes)
    self._update_node_cache()