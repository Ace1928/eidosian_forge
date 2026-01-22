import logging
from typing import Any, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
Call to Modal endpoint.