from typing import Any, Dict, List, Optional
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
Use praw to search Reddit and return a list of dictionaries,
        one for each post.
        