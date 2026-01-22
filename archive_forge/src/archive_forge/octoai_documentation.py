from typing import Dict
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.utils.openai import is_openai_v1
Validate that api key and python package exists in environment.