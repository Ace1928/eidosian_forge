from __future__ import annotations
import warnings
from abc import ABC
from string import Formatter
from typing import Any, Callable, Dict, List, Set, Tuple, Type
import langchain_core.utils.mustache as mustache
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, create_model
from langchain_core.utils import get_colored_text
from langchain_core.utils.formatting import formatter
from langchain_core.utils.interactive_env import is_interactive_env
def mustache_template_vars(template: str) -> Set[str]:
    """Get the variables from a mustache template."""
    vars: Set[str] = set()
    in_section = False
    for type, key in mustache.tokenize(template):
        if type == 'end':
            in_section = False
        elif in_section:
            continue
        elif type in ('variable', 'section') and key != '.':
            vars.add(key.split('.')[0])
            if type == 'section':
                in_section = True
    return vars