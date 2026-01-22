import copy
import importlib
import json
import pprint
import textwrap
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Type, TYPE_CHECKING, Union
import gitlab
from gitlab import types as g_types
from gitlab.exceptions import GitlabParsingError
from .client import Gitlab, GitlabList
@property
def encoded_id(self) -> Optional[Union[int, str]]:
    """Ensure that the ID is url-encoded so that it can be safely used in a URL
        path"""
    obj_id = self.get_id()
    if isinstance(obj_id, str):
        obj_id = gitlab.utils.EncodedId(obj_id)
    return obj_id