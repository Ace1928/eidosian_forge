import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def update_task(self, query: str) -> Dict:
    """
        Update an attribute of a specified task.
        """
    query_dict, error = load_query(query, fault_tolerant=True)
    if query_dict is None:
        return {'Error': error}
    url = f'{DEFAULT_URL}/task/{query_dict['task_id']}'
    params = {'custom_task_ids': 'true', 'team_id': self.team_id, 'include_subtasks': 'true'}
    headers = self.get_headers()
    payload = {query_dict['attribute_name']: query_dict['value']}
    response = requests.put(url, headers=headers, params=params, json=payload)
    return {'response': response}