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
@mock.patch.dict(os.environ, clear='CIRQ_TESTING')
def test_not_yet_serializable_no_superfluous(mod_spec: ModuleJsonTestSpec):
    names = set(mod_spec.get_all_names())
    missing_names = set(mod_spec.not_yet_serializable).difference(names)
    assert len(missing_names) == 0, f'Defined as Not yet serializable, but missing from {mod_spec}: \n{missing_names}'