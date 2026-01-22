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
def assert_repr_and_json_test_data_agree(mod_spec: ModuleJsonTestSpec, repr_path: pathlib.Path, json_path: pathlib.Path, inward_only: bool, deprecation_deadline: Optional[str]):
    if not repr_path.exists() and (not json_path.exists()):
        return
    rel_repr_path = f'{repr_path.relative_to(REPO_ROOT)}'
    rel_json_path = f'{json_path.relative_to(REPO_ROOT)}'
    try:
        json_from_file = json_path.read_text()
        ctx_manager = cirq.testing.assert_deprecated(deadline=deprecation_deadline, count=None) if deprecation_deadline else contextlib.suppress()
        with ctx_manager:
            json_obj = cirq.read_json(json_text=json_from_file)
    except ValueError as ex:
        if 'Could not resolve type' in str(ex):
            mod_path = mod_spec.name.replace('.', '/')
            rel_resolver_cache_path = f'{mod_path}/json_resolver_cache.py'
            pytest.fail(f"{rel_json_path} can't be parsed to JSON.\nMaybe an entry is missing from the  `_class_resolver_dictionary` method in {rel_resolver_cache_path}?")
        else:
            raise ValueError(f'deprecation: {deprecation_deadline} - got error: {ex}')
    except AssertionError as ex:
        raise ex
    except Exception as ex:
        raise IOError(f'Failed to parse test json data from {rel_json_path}.') from ex
    try:
        repr_obj = _eval_repr_data_file(repr_path, deprecation_deadline)
    except Exception as ex:
        raise IOError(f'Failed to parse test repr data from {rel_repr_path}.') from ex
    assert proper_eq(json_obj, repr_obj), f'The json data from {rel_json_path} did not parse into an object equivalent to the repr data from {rel_repr_path}.\n\njson object: {json_obj!r}\nrepr object: {repr_obj!r}\n'
    if not inward_only:
        json_from_cirq = cirq.to_json(repr_obj)
        json_from_cirq_obj = json.loads(json_from_cirq)
        json_from_file_obj = json.loads(json_from_file)
        assert json_from_cirq_obj == json_from_file_obj, f'The json produced by cirq no longer agrees with the json in the {rel_json_path} test data file.\n\nYou must either fix the cirq code to continue to produce the same output, or you must move the old test data to {rel_json_path}_inward and create a fresh {rel_json_path} file.\n\ntest data json:\n{json_from_file}\n\ncirq produced json:\n{json_from_cirq}\n'