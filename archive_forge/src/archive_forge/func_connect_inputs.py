import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
def connect_inputs(self):
    """Connect stereotyped inputs to the input IdentityInterface."""
    self.wf.connect(self.subj_node, 'subject_id', self.grab_node, 'subject_id')
    if hasattr(self.in_node.inputs, 'subject_id'):
        self.wf.connect(self.subj_node, 'subject_id', self.in_node, 'subject_id')
    grabbed = self.grab_node.outputs.get()
    inputs = self.in_node.inputs.get()
    for field in grabbed:
        if field in inputs:
            self.wf.connect(self.grab_node, field, self.in_node, field)