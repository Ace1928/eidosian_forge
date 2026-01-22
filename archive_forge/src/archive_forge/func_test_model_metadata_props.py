import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_model_metadata_props(self) -> None:
    graph = helper.make_graph([], 'my graph', [], [])
    model_def = helper.make_model(graph, doc_string='test')
    helper.set_model_props(model_def, {'Title': 'my graph', 'Keywords': 'test;graph'})
    checker.check_model(model_def)
    helper.set_model_props(model_def, {'Title': 'my graph', 'Keywords': 'test;graph'})
    checker.check_model(model_def)
    dupe = model_def.metadata_props.add()
    dupe.key = 'Title'
    dupe.value = 'Other'
    self.assertRaises(checker.ValidationError, checker.check_model, model_def)