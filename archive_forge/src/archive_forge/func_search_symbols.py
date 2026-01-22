from typing import Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def search_symbols(self, keywords: str) -> Dict[str, Any]:
    """Make a request to the AlphaVantage API to search for symbols."""
    response = requests.get('https://www.alphavantage.co/query/', params={'function': 'SYMBOL_SEARCH', 'keywords': keywords, 'apikey': self.alphavantage_api_key})
    response.raise_for_status()
    data = response.json()
    if 'Error Message' in data:
        raise ValueError(f'API Error: {data['Error Message']}')
    return data