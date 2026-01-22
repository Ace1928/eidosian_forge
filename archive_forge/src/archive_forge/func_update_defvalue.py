import ast
import inspect
import sys
from inspect import Parameter
from typing import Any, Dict, List, Optional
import sphinx
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.pycode.ast import parse as ast_parse
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
    """Update defvalue info of *obj* using type_comments."""
    if not app.config.autodoc_preserve_defaults:
        return
    try:
        lines = inspect.getsource(obj).splitlines()
        if lines[0].startswith((' ', '\\t')):
            lines.insert(0, '')
    except (OSError, TypeError):
        lines = []
    try:
        function = get_function_def(obj)
        if function.args.defaults or function.args.kw_defaults:
            sig = inspect.signature(obj)
            defaults = list(function.args.defaults)
            kw_defaults = list(function.args.kw_defaults)
            parameters = list(sig.parameters.values())
            for i, param in enumerate(parameters):
                if param.default is param.empty:
                    if param.kind == param.KEYWORD_ONLY:
                        kw_defaults.pop(0)
                elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                    default = defaults.pop(0)
                    value = get_default_value(lines, default)
                    if value is None:
                        value = ast_unparse(default)
                    parameters[i] = param.replace(default=DefaultValue(value))
                else:
                    default = kw_defaults.pop(0)
                    value = get_default_value(lines, default)
                    if value is None:
                        value = ast_unparse(default)
                    parameters[i] = param.replace(default=DefaultValue(value))
            if bound_method and inspect.ismethod(obj):
                cls = inspect.Parameter('cls', Parameter.POSITIONAL_OR_KEYWORD)
                parameters.insert(0, cls)
            sig = sig.replace(parameters=parameters)
            if bound_method and inspect.ismethod(obj):
                obj.__dict__['__signature__'] = sig
            else:
                obj.__signature__ = sig
    except (AttributeError, TypeError):
        pass
    except NotImplementedError as exc:
        logger.warning(__('Failed to parse a default argument value for %r: %s'), obj, exc)