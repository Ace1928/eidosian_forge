from __future__ import annotations
import os
import pathlib
import site
import stat
import sys
from collections import OrderedDict
from contextlib import contextmanager
import pytest
import yaml
import dask.config
from dask.config import (
def test_update_defaults():
    defaults = [{'a': 1, 'b': {'c': 1}}, {'a': 2, 'b': {'d': 2}}]
    current = {'a': 2, 'b': {'c': 1, 'd': 3}, 'extra': 0}
    new = {'a': 0, 'b': {'c': 0, 'd': 0}, 'new-extra': 0}
    update_defaults(new, current, defaults=defaults)
    assert defaults == [{'a': 1, 'b': {'c': 1}}, {'a': 2, 'b': {'d': 2}}, {'a': 0, 'b': {'c': 0, 'd': 0}, 'new-extra': 0}]
    assert current == {'a': 0, 'b': {'c': 0, 'd': 3}, 'extra': 0, 'new-extra': 0}