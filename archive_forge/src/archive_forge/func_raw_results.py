import json
from typing import Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
def raw_results(self, query: str, max_results: Optional[int]=5, search_depth: Optional[str]='advanced', include_domains: Optional[List[str]]=[], exclude_domains: Optional[List[str]]=[], include_answer: Optional[bool]=False, include_raw_content: Optional[bool]=False, include_images: Optional[bool]=False) -> Dict:
    params = {'api_key': self.tavily_api_key.get_secret_value(), 'query': query, 'max_results': max_results, 'search_depth': search_depth, 'include_domains': include_domains, 'exclude_domains': exclude_domains, 'include_answer': include_answer, 'include_raw_content': include_raw_content, 'include_images': include_images}
    response = requests.post(f'{TAVILY_API_URL}/search', json=params)
    response.raise_for_status()
    return response.json()