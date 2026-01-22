import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def update_task_assignees(self, query: str) -> Dict:
    """
        Add or remove assignees of a specified task.
        """
    query_dict, error = load_query(query, fault_tolerant=True)
    if query_dict is None:
        return {'Error': error}
    for user in query_dict['users']:
        if not isinstance(user, int):
            return {'Error': f'All users must be integers, not strings!\n"Got user {user} if type {type(user)}'}
    url = f'{DEFAULT_URL}/task/{query_dict['task_id']}'
    headers = self.get_headers()
    if query_dict['operation'] == 'add':
        assigne_payload = {'add': query_dict['users'], 'rem': []}
    elif query_dict['operation'] == 'rem':
        assigne_payload = {'add': [], 'rem': query_dict['users']}
    else:
        raise ValueError(f'Invalid operation ({query_dict['operation']}). ', "Valid options ['add', 'rem'].")
    params = {'custom_task_ids': 'true', 'team_id': self.team_id, 'include_subtasks': 'true'}
    payload = {'assignees': assigne_payload}
    response = requests.put(url, headers=headers, params=params, json=payload)
    return {'response': response}