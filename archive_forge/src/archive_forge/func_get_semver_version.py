import json
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union
import tornado
from jupyterlab_server.translation_utils import translator
from traitlets import Enum
from traitlets.config import Configurable, LoggingConfigurable
from jupyterlab.commands import (
@staticmethod
def get_semver_version(version: str) -> str:
    """Convert a Python version to Semver version.

        It:

        - drops ``.devN`` and ``.postN``
        - converts ``aN``, ``bN`` and ``rcN`` to ``-alpha.N``, ``-beta.N``, ``-rc.N`` respectively

        Args:
            version: Version to convert
        Returns
            Semver compatible version
        """
    return re.sub('(a|b|rc)(\\d+)$', lambda m: f'{PYTHON_TO_SEMVER[m.group(1)]}{m.group(2)}', re.subn('\\.(dev|post)\\d+', '', version)[0])