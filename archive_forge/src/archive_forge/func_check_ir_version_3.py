import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def check_ir_version_3(g: GraphProto) -> None:
    checker.check_graph(g, ctx, lex_ctx)