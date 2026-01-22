import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name  # type: ignore
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.utils import is_valid_field
def to_argument(self, info: TypeInfo, typed: bool, force_optional: bool, use_alias: bool) -> Argument:
    if typed and info[self.name].type is not None:
        type_annotation = info[self.name].type
    else:
        type_annotation = AnyType(TypeOfAny.explicit)
    return Argument(variable=self.to_var(info, use_alias), type_annotation=type_annotation, initializer=None, kind=ARG_NAMED_OPT if force_optional or not self.is_required else ARG_NAMED)