from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts.chat import (
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
@root_validator()
def template_is_valid(cls, values: Dict) -> Dict:
    """Check that prefix, suffix, and input variables are consistent."""
    if values['validate_template']:
        check_valid_template(values['prefix'] + values['suffix'], values['template_format'], values['input_variables'] + list(values['partial_variables']))
    elif values.get('template_format'):
        values['input_variables'] = [var for var in get_template_variables(values['prefix'] + values['suffix'], values['template_format']) if var not in values['partial_variables']]
    return values