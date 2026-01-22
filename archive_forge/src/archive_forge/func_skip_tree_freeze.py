from typing import cast
import pytest
import cirq
def skip_tree_freeze(root):
    return cirq.freeze_op_tree(cirq.transform_op_tree(root, iter_transformation=skip_first))