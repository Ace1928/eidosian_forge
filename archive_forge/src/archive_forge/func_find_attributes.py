import time
from sys import platform
from typing import (
def find_attributes(attributes: Dict[int, Any], keys: List[str]) -> Dict[str, str]:
    values = {}
    for [key_index, value_index] in zip(*(iter(attributes),) * 2):
        if value_index < 0:
            continue
        key = strings[key_index]
        value = strings[value_index]
        if key in keys:
            values[key] = value
            keys.remove(key)
            if not keys:
                return values
    return values