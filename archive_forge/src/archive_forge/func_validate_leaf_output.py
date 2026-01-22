import collections
import importlib.util
import json
import os
import tempfile
from typing import Any, Callable, Dict, Type
import numpy as np
import xgboost as xgb
from xgboost._typing import ArrayLike
def validate_leaf_output(leaf: np.ndarray, num_parallel_tree: int) -> None:
    """Validate output for predict leaf tests."""
    for i in range(leaf.shape[0]):
        for j in range(leaf.shape[1]):
            for k in range(leaf.shape[2]):
                tree_group = leaf[i, j, k, :]
                assert tree_group.shape[0] == num_parallel_tree
                assert np.all(tree_group == tree_group[0])