import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def get_authorized_teams(self) -> Dict[Any, Any]:
    """Get all teams for the user."""
    url = f'{DEFAULT_URL}/team'
    response = requests.get(url, headers=self.get_headers())
    data = response.json()
    parsed_teams = self.attempt_parse_teams(data)
    return parsed_teams