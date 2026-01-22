import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
def make_subject_source(subject_list):
    """Generate a source node with iterables over a subject_id list."""
    return Node(IdentityInterface(fields=['subject_id']), iterables=('subject_id', subject_list), overwrite=True, name='subj_source')