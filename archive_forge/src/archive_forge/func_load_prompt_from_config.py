import json
import logging
from pathlib import Path
from typing import Callable, Dict, Union
import yaml
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
def load_prompt_from_config(config: dict) -> BasePromptTemplate:
    """Load prompt from Config Dict."""
    if '_type' not in config:
        logger.warning('No `_type` key found, defaulting to `prompt`.')
    config_type = config.pop('_type', 'prompt')
    if config_type not in type_to_loader_dict:
        raise ValueError(f'Loading {config_type} prompt not supported')
    prompt_loader = type_to_loader_dict[config_type]
    return prompt_loader(config)