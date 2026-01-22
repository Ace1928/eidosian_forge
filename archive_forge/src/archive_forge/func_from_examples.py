from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from langchain_core.prompts.string import (
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.runnables.config import RunnableConfig
@classmethod
def from_examples(cls, examples: List[str], suffix: str, input_variables: List[str], example_separator: str='\n\n', prefix: str='', **kwargs: Any) -> PromptTemplate:
    """Take examples in list format with prefix and suffix to create a prompt.

        Intended to be used as a way to dynamically create a prompt from examples.

        Args:
            examples: List of examples to use in the prompt.
            suffix: String to go after the list of examples. Should generally
                set up the user's input.
            input_variables: A list of variable names the final prompt template
                will expect.
            example_separator: The separator to use in between examples. Defaults
                to two new line characters.
            prefix: String that should go before any examples. Generally includes
                examples. Default to an empty string.

        Returns:
            The final prompt generated.
        """
    template = example_separator.join([prefix, *examples, suffix])
    return cls(input_variables=input_variables, template=template, **kwargs)