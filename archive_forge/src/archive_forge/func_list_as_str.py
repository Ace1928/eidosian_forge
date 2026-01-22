import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests import Request, Session
def list_as_str(self) -> str:
    """Same as list, but returns a stringified version of the JSON for
        insertting back into an LLM."""
    actions = self.list()
    return json.dumps(actions)