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
def test_op_roundtrip():
    q = cirq.LineQubit(5)
    op1 = cirq.rx(0.123).on(q)
    assert_json_roundtrip_works(op1, text_should_be='{\n  "cirq_type": "GateOperation",\n  "gate": {\n    "cirq_type": "Rx",\n    "rads": 0.123\n  },\n  "qubits": [\n    {\n      "cirq_type": "LineQubit",\n      "x": 5\n    }\n  ]\n}')