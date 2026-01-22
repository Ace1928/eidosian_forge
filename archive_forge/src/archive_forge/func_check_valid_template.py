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
def check_valid_template(template: str, template_format: str, input_variables: List[str]) -> None:
    """Check that template string is valid.

    Args:
        template: The template string.
        template_format: The template format. Should be one of "f-string" or "jinja2".
        input_variables: The input variables.

    Raises:
        ValueError: If the template format is not supported.
    """
    try:
        validator_func = DEFAULT_VALIDATOR_MAPPING[template_format]
    except KeyError as exc:
        raise ValueError(f'Invalid template format {template_format!r}, should be one of {list(DEFAULT_FORMATTER_MAPPING)}.') from exc
    try:
        validator_func(template, input_variables)
    except (KeyError, IndexError) as exc:
        raise ValueError(f'Invalid prompt schema; check for mismatched or missing input parameters from {input_variables}.') from exc