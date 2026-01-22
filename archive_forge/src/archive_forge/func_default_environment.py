import operator
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ._parser import (
from ._parser import (
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name
def default_environment() -> Dict[str, str]:
    iver = format_full_version(sys.implementation.version)
    implementation_name = sys.implementation.name
    return {'implementation_name': implementation_name, 'implementation_version': iver, 'os_name': os.name, 'platform_machine': platform.machine(), 'platform_release': platform.release(), 'platform_system': platform.system(), 'platform_version': platform.version(), 'python_full_version': platform.python_version(), 'platform_python_implementation': platform.python_implementation(), 'python_version': '.'.join(platform.python_version_tuple()[:2]), 'sys_platform': sys.platform}