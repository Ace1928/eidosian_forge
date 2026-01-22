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
def test_line_qubit_roundtrip():
    q1 = cirq.LineQubit(12)
    assert_json_roundtrip_works(q1, text_should_be='{\n  "cirq_type": "LineQubit",\n  "x": 12\n}')