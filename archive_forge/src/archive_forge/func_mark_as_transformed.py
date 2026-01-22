import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
def mark_as_transformed(self, node: 'Node'):
    """
        Marks a node as transformed by this transformation.

        Args:
            node (`torch.fx.Node`):
                The node to mark as transformed.
        """
    node_transformations = getattr(node, 'transformations', set())
    node_transformations.add(self.signature)
    node.transformations = node_transformations