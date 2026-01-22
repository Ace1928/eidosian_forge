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
def test_internal_serializer_types():
    sbki = SBKImpl('test_key')
    key = 1
    test_key = json_serialization._SerializedKey(key)
    test_context = json_serialization._SerializedContext(sbki, 1)
    test_serialization = json_serialization._ContextualSerialization(sbki)
    key_json = test_key._json_dict_()
    with pytest.raises(TypeError, match='_from_json_dict_'):
        _ = json_serialization._SerializedKey._from_json_dict_(**key_json)
    context_json = test_context._json_dict_()
    with pytest.raises(TypeError, match='_from_json_dict_'):
        _ = json_serialization._SerializedContext._from_json_dict_(**context_json)
    serialization_json = test_serialization._json_dict_()
    with pytest.raises(TypeError, match='_from_json_dict_'):
        _ = json_serialization._ContextualSerialization._from_json_dict_(**serialization_json)