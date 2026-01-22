import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
@classmethod
def get_access_code_url(cls, oauth_client_id: str, redirect_uri: str='https://google.com') -> str:
    """Get the URL to get an access code."""
    url = f'https://app.clickup.com/api?client_id={oauth_client_id}'
    return f'{url}&redirect_uri={redirect_uri}'