from __future__ import annotations
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import os
import re
import subprocess
import sys
import unittest
from attrs import field, frozen
from referencing import Registry
import referencing.jsonschema
from jsonschema.validators import _VALIDATORS
import jsonschema
def to_unittest_testcase(self, *groups, **kwargs):
    name = kwargs.pop('name', 'Test' + self.name.title().replace('-', ''))
    methods = {method.__name__: method for method in (test.to_unittest_method(**kwargs) for group in groups for case in group for test in case.tests)}
    cls = type(name, (unittest.TestCase,), methods)
    with suppress(Exception):
        cls.__module__ = _someone_save_us_the_module_of_the_caller()
    return cls