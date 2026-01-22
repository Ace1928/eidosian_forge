import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import root_validator
def default_guardrail_violation_handler(violation: dict) -> str:
    """Default guardrail violation handler.

    Args:
        violation (dict): The violation dictionary.

    Returns:
        str: The canned response.
    """
    if violation.get('canned_response'):
        return violation['canned_response']
    guardrail_name = f'Guardrail {violation.get('offending_guardrail')}' if violation.get('offending_guardrail') else 'A guardrail'
    raise ValueError(f'{guardrail_name} was violated without a proper guardrail violation handler.')