from typing import Any, List
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
def parse_to_str(self, details: dict) -> str:
    """Parse the details result."""
    result = ''
    for key, value in details.items():
        result += 'The ' + str(key) + ' is: ' + str(value) + '\n'
    return result