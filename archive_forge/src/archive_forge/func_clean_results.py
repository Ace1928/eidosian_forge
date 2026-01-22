import json
from typing import Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
def clean_results(self, results: List[Dict]) -> List[Dict]:
    """Clean results from Tavily Search API."""
    clean_results = []
    for result in results:
        clean_results.append({'url': result['url'], 'content': result['content']})
    return clean_results