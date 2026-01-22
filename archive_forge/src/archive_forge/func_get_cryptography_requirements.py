from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
def get_cryptography_requirements(python: PythonConfig) -> list[str]:
    """
    Return the correct cryptography and pyopenssl requirements for the given python version.
    The version of cryptography installed depends on the python version and openssl version.
    """
    openssl_version = get_openssl_version(python)
    if openssl_version and openssl_version < (1, 1, 0):
        cryptography = 'cryptography < 3.2'
        pyopenssl = 'pyopenssl < 20.0.0'
    else:
        cryptography = 'cryptography'
        pyopenssl = ''
    requirements = [cryptography, pyopenssl]
    requirements = [requirement for requirement in requirements if requirement]
    return requirements