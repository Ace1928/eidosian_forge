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
@pytest.mark.parametrize('mod_spec, abs_path', [(m, abs_path) for m in MODULE_TEST_SPECS for abs_path in m.all_test_data_keys()])
def test_json_and_repr_data(mod_spec: ModuleJsonTestSpec, abs_path: str):
    assert_repr_and_json_test_data_agree(mod_spec=mod_spec, repr_path=pathlib.Path(f'{abs_path}.repr'), json_path=pathlib.Path(f'{abs_path}.json'), inward_only=False, deprecation_deadline=mod_spec.deprecated.get(os.path.basename(abs_path)))
    assert_repr_and_json_test_data_agree(mod_spec=mod_spec, repr_path=pathlib.Path(f'{abs_path}.repr_inward'), json_path=pathlib.Path(f'{abs_path}.json_inward'), inward_only=True, deprecation_deadline=mod_spec.deprecated.get(os.path.basename(abs_path)))