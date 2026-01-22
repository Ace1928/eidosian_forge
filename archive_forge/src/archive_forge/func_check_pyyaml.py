from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
def check_pyyaml(python: PythonConfig, required: bool=True, quiet: bool=False) -> t.Optional[bool]:
    """
    Return True if PyYAML has libyaml support, False if it does not and None if it was not found.
    The result is cached if True or required.
    """
    try:
        return CHECK_YAML_VERSIONS[python.path]
    except KeyError:
        pass
    state = yamlcheck(python)
    if state is not None or required:
        CHECK_YAML_VERSIONS[python.path] = state
    if not quiet:
        if state is None:
            if required:
                display.warning('PyYAML is not installed for interpreter: %s' % python.path)
        elif not state:
            display.warning('PyYAML will be slow due to installation without libyaml support for interpreter: %s' % python.path)
    return state