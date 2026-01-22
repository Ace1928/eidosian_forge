import base64
from abc import ABC
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Literal, Optional, Union
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
def get_file_content_by_path(self, path: str) -> str:
    base_url = f'{self.github_api_url}/repos/{self.repo}/contents/{path}'
    response = requests.get(base_url, headers=self.headers)
    response.raise_for_status()
    if isinstance(response.json(), dict):
        content_encoded = response.json()['content']
        return base64.b64decode(content_encoded).decode('utf-8')
    return ''