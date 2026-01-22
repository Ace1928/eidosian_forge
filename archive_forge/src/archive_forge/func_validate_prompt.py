import json
from json import JSONDecodeError
from typing import Any, List, Optional, Sequence, Tuple, Union
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain.agents import BaseMultiActionAgent
from langchain.agents.format_scratchpad.openai_functions import (
@root_validator
def validate_prompt(cls, values: dict) -> dict:
    prompt: BasePromptTemplate = values['prompt']
    if 'agent_scratchpad' not in prompt.input_variables:
        raise ValueError(f'`agent_scratchpad` should be one of the variables in the prompt, got {prompt.input_variables}')
    return values