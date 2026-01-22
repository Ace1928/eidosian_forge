import filecmp
import importlib
import os
import re
import shutil
import signal
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import try_to_load_from_cache
from .utils import (
def get_relative_imports(module_file: Union[str, os.PathLike]) -> List[str]:
    """
    Get the list of modules that are relatively imported in a module file.

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.

    Returns:
        `List[str]`: The list of relative imports in the module.
    """
    with open(module_file, 'r', encoding='utf-8') as f:
        content = f.read()
    relative_imports = re.findall('^\\s*import\\s+\\.(\\S+)\\s*$', content, flags=re.MULTILINE)
    relative_imports += re.findall('^\\s*from\\s+\\.(\\S+)\\s+import', content, flags=re.MULTILINE)
    return list(set(relative_imports))