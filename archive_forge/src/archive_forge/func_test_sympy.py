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
def test_sympy():
    assert_json_roundtrip_works(sympy.Symbol('theta'))
    assert_json_roundtrip_works(sympy.Integer(5))
    assert_json_roundtrip_works(sympy.Rational(2, 3))
    assert_json_roundtrip_works(sympy.Float(1.1))
    s = sympy.Symbol('s')
    t = sympy.Symbol('t')
    assert_json_roundtrip_works(t + s)
    assert_json_roundtrip_works(t * s)
    assert_json_roundtrip_works(t / s)
    assert_json_roundtrip_works(t - s)
    assert_json_roundtrip_works(t ** s)
    assert_json_roundtrip_works(t * 2)
    assert_json_roundtrip_works(4 * t + 3 * s + 2)
    assert_json_roundtrip_works(sympy.pi)
    assert_json_roundtrip_works(sympy.E)
    assert_json_roundtrip_works(sympy.EulerGamma)