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
def optional_cases_of(self, name: str) -> Iterable[_Case]:
    return self._cases_in(paths=[self._path / 'optional' / f'{name}.json'])