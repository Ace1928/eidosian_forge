from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts.chat import (
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
@root_validator(pre=True)
def check_examples_and_selector(cls, values: Dict) -> Dict:
    """Check that one and only one of examples/example_selector are provided."""
    examples = values.get('examples', None)
    example_selector = values.get('example_selector', None)
    if examples and example_selector:
        raise ValueError("Only one of 'examples' and 'example_selector' should be provided")
    if examples is None and example_selector is None:
        raise ValueError("One of 'examples' and 'example_selector' should be provided")
    return values