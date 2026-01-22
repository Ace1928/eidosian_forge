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
def jinja2_formatter(template: str, **kwargs: Any) -> str:
    """Format a template using jinja2.

    *Security warning*: As of LangChain 0.0.329, this method uses Jinja2's
        SandboxedEnvironment by default. However, this sand-boxing should
        be treated as a best-effort approach rather than a guarantee of security.
        Do not accept jinja2 templates from untrusted sources as they may lead
        to arbitrary Python code execution.

        https://jinja.palletsprojects.com/en/3.1.x/sandbox/
    """
    try:
        from jinja2.sandbox import SandboxedEnvironment
    except ImportError:
        raise ImportError('jinja2 not installed, which is needed to use the jinja2_formatter. Please install it with `pip install jinja2`.Please be cautious when using jinja2 templates. Do not expand jinja2 templates using unverified or user-controlled inputs as that can result in arbitrary Python code execution.')
    return SandboxedEnvironment().from_string(template).render(**kwargs)