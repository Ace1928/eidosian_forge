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
@pytest.mark.parametrize('mod_spec', MODULE_TEST_SPECS, ids=repr)
def test_mutually_exclusive_lists(mod_spec: ModuleJsonTestSpec):
    common = set(mod_spec.should_not_be_serialized) & set(mod_spec.not_yet_serializable)
    assert len(common) == 0, f"Defined in both {mod_spec.name} 'Not yet serializable'  and 'Should not be serialized' lists: {common}"