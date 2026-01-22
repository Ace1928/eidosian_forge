import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
def list_out_file(fname):
    """Return a _list_outputs function for a single out_file."""

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = op.abspath(fname)
        return outputs
    return _list_outputs