import re
from typing import Callable, Dict, List
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, validator
@validator('example_text_lengths', always=True)
def calculate_example_text_lengths(cls, v: List[int], values: Dict) -> List[int]:
    """Calculate text lengths if they don't exist."""
    if v:
        return v
    example_prompt = values['example_prompt']
    get_text_length = values['get_text_length']
    string_examples = [example_prompt.format(**eg) for eg in values['examples']]
    return [get_text_length(eg) for eg in string_examples]