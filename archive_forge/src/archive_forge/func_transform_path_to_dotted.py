import os
import re
from pathlib import Path
from importlib.machinery import all_suffixes
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ContextualizedNode
from jedi.inference.helpers import is_string, get_str_or_none
from jedi.parser_utils import get_cached_code_lines
from jedi.file_io import FileIO
from jedi import settings
from jedi import debug
def transform_path_to_dotted(sys_path, module_path):
    """
    Returns the dotted path inside a sys.path as a list of names. e.g.

    >>> transform_path_to_dotted([str(Path("/foo").absolute())], Path('/foo/bar/baz.py').absolute())
    (('bar', 'baz'), False)

    Returns (None, False) if the path doesn't really resolve to anything.
    The second return part is if it is a package.
    """
    module_path = remove_python_path_suffix(module_path)
    if module_path.name.startswith('.'):
        return (None, False)
    is_package = module_path.name == '__init__'
    if is_package:
        module_path = module_path.parent

    def iter_potential_solutions():
        for p in sys_path:
            if str(module_path).startswith(p):
                rest = str(module_path)[len(p):]
                if rest.startswith(os.path.sep) or rest.startswith('/'):
                    rest = rest[1:]
                if rest:
                    split = rest.split(os.path.sep)
                    if not all(split):
                        return
                    yield tuple((re.sub('-stubs$', '', s) for s in split))
    potential_solutions = tuple(iter_potential_solutions())
    if not potential_solutions:
        return (None, False)
    return (sorted(potential_solutions, key=lambda p: len(p))[0], is_package)