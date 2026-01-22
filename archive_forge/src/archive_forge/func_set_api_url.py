from __future__ import annotations
import logging
from typing import (
import requests
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
@root_validator(pre=True, allow_reuse=True)
def set_api_url(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    if 'api_url' not in values:
        host = values['host']
        group_id = values['group_id']
        api_url = f'{host}/v1/text/chatcompletion?GroupId={group_id}'
        values['api_url'] = api_url
    return values