import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence,
from docutils.statemachine import StringList
import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import (get_class_members, get_object_members, import_module,
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
from sphinx.util.typing import OptionSpec, get_type_hints, restify
from sphinx.util.typing import stringify as stringify_typehint
def sort_members(self, documenters: List[Tuple['Documenter', bool]], order: str) -> List[Tuple['Documenter', bool]]:
    if order == 'bysource' and self.__all__:
        documenters.sort(key=lambda e: e[0].name)

        def keyfunc(entry: Tuple[Documenter, bool]) -> int:
            name = entry[0].name.split('::')[1]
            if self.__all__ and name in self.__all__:
                return self.__all__.index(name)
            else:
                return len(self.__all__)
        documenters.sort(key=keyfunc)
        return documenters
    else:
        return super().sort_members(documenters, order)