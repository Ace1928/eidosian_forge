import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def fetch_team_id(access_token: str) -> Optional[int]:
    """Fetch the team id."""
    url = f'{DEFAULT_URL}/team'
    data = fetch_data(url, access_token)
    return fetch_first_id(data, 'teams')