import json
from typing import Any, Callable, Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
Execute a GraphQL query and return the results.