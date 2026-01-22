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
@pytest.mark.parametrize('mod_spec,cirq_obj_name,cls', _list_public_classes_for_tested_modules())
def test_type_serialization(mod_spec: ModuleJsonTestSpec, cirq_obj_name: str, cls):
    if cirq_obj_name in mod_spec.tested_elsewhere:
        pytest.skip('Tested elsewhere.')
    if cirq_obj_name in mod_spec.not_yet_serializable:
        return pytest.xfail(reason='Not serializable (yet)')
    if cls is None:
        pytest.skip(f'No serialization for None-mapped type: {cirq_obj_name}')
    try:
        typename = cirq.json_cirq_type(cls)
    except ValueError as e:
        pytest.skip(f'No serialization for non-Cirq type: {str(e)}')

    def custom_resolver(name):
        if name == 'SerializableTypeObject':
            return SerializableTypeObject
    sto = SerializableTypeObject(cls)
    test_resolvers = [custom_resolver] + cirq.DEFAULT_RESOLVERS
    expected_json = f'{{\n  "cirq_type": "SerializableTypeObject",\n  "test_type": "{typename}"\n}}'
    assert cirq.to_json(sto) == expected_json
    assert cirq.read_json(json_text=expected_json, resolvers=test_resolvers) == sto
    assert_json_roundtrip_works(sto, resolvers=test_resolvers)