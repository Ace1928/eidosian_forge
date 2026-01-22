import json
from typing import Dict, Iterator, List, Optional
from urllib.parse import quote
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
Run query through Merriam-Webster API and return a formatted result.