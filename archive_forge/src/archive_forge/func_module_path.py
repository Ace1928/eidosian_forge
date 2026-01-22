import re
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from jedi import settings
from jedi import debug
from jedi.inference.utils import unite
from jedi.cache import memoize_method
from jedi.inference.compiled.mixed import MixedName
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.conversion import convert_names, convert_values
from jedi.inference.base_value import ValueSet, HasNoContext
from jedi.api.keywords import KeywordName
from jedi.api import completion_cache
from jedi.api.helpers import filter_follow_imports
@property
def module_path(self) -> Optional[Path]:
    """
        Shows the file path of a module. e.g. ``/usr/lib/python3.9/os.py``
        """
    module = self._get_module_context()
    if module.is_stub() or not module.is_compiled():
        path: Optional[Path] = self._get_module_context().py__file__()
        return path
    return None