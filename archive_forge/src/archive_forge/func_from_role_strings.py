from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from langchain_core._api import deprecated
from langchain_core.load import Serializable
from langchain_core.messages import (
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompt_values import ChatPromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import StringPromptTemplate, get_template_variables
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env
@classmethod
@deprecated('0.0.1', alternative='from_messages classmethod', pending=True)
def from_role_strings(cls, string_messages: List[Tuple[str, str]]) -> ChatPromptTemplate:
    """Create a chat prompt template from a list of (role, template) tuples.

        Args:
            string_messages: list of (role, template) tuples.

        Returns:
            a chat prompt template
        """
    return cls(messages=[ChatMessagePromptTemplate.from_template(template, role=role) for role, template in string_messages])