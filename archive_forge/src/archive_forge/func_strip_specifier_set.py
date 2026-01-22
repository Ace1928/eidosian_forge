import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import (
from mypy_extensions import mypyc_attr
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from black.handle_ipynb_magics import jupyter_dependencies_are_installed
from black.mode import TargetVersion
from black.output import err
from black.report import Report
def strip_specifier_set(specifier_set: SpecifierSet) -> SpecifierSet:
    """Strip minor versions for some specifiers in the specifier set.

    For background on version specifiers, see PEP 440:
    https://peps.python.org/pep-0440/#version-specifiers
    """
    specifiers = []
    for s in specifier_set:
        if '*' in str(s):
            specifiers.append(s)
        elif s.operator in ['~=', '==', '>=', '===']:
            version = Version(s.version)
            stripped = Specifier(f'{s.operator}{version.major}.{version.minor}')
            specifiers.append(stripped)
        elif s.operator == '>':
            version = Version(s.version)
            if len(version.release) > 2:
                s = Specifier(f'>={version.major}.{version.minor}')
            specifiers.append(s)
        else:
            specifiers.append(s)
    return SpecifierSet(','.join((str(s) for s in specifiers)))