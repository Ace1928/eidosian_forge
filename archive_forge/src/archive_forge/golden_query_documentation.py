import json
from typing import Dict, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
Run query through Golden Query API and return the JSON raw result.