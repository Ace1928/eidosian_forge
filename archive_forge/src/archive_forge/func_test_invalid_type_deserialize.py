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
def test_invalid_type_deserialize():

    def custom_resolver(name):
        if name == 'SerializableTypeObject':
            return SerializableTypeObject
    test_resolvers = [custom_resolver] + cirq.DEFAULT_RESOLVERS
    invalid_json = '{\n  "cirq_type": "SerializableTypeObject",\n  "test_type": "bad_type"\n}'
    with pytest.raises(ValueError, match='Could not resolve type'):
        _ = cirq.read_json(json_text=invalid_json, resolvers=test_resolvers)
    factory_json = '{\n  "cirq_type": "SerializableTypeObject",\n  "test_type": "sympy.Add"\n}'
    with pytest.raises(ValueError, match='maps to a factory method'):
        _ = cirq.read_json(json_text=factory_json, resolvers=test_resolvers)