from typing import Any, Dict, List, Tuple
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import BaseChatPromptTemplate
from langchain_core.pydantic_v1 import root_validator
@root_validator(pre=True)
def get_input_variables(cls, values: Dict) -> Dict:
    """Get input variables."""
    created_variables = set()
    all_variables = set()
    for k, prompt in values['pipeline_prompts']:
        created_variables.add(k)
        all_variables.update(prompt.input_variables)
    values['input_variables'] = list(all_variables.difference(created_variables))
    return values