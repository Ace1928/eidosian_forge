import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def fetch_list_id(space_id: int, folder_id: int, access_token: str) -> Optional[int]:
    """Fetch the list id."""
    if folder_id:
        url = f'{DEFAULT_URL}/folder/{folder_id}/list'
    else:
        url = f'{DEFAULT_URL}/space/{space_id}/list'
    data = fetch_data(url, access_token, query={'archived': 'false'})
    if folder_id and 'id' in data:
        return data['id']
    else:
        return fetch_first_id(data, 'lists')