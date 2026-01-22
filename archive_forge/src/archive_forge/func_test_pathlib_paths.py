import contextlib
import dataclasses
import datetime
import importlib
import io
import json
import os
import pathlib
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Type
from unittest import mock
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import sympy
import cirq
from cirq._compat import proper_eq
from cirq.protocols import json_serialization
from cirq.testing.json import ModuleJsonTestSpec, spec_for, assert_json_roundtrip_works
def test_pathlib_paths(tmpdir):
    path = pathlib.Path(tmpdir) / 'op.json'
    cirq.to_json(cirq.X, path)
    assert cirq.read_json(path) == cirq.X
    gzip_path = pathlib.Path(tmpdir) / 'op.gz'
    cirq.to_json_gzip(cirq.X, gzip_path)
    assert cirq.read_json_gzip(gzip_path) == cirq.X