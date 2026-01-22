import importlib
import sys
def is_transformers_greater_than(version: str) -> bool:
    _transformers_version = importlib.metadata.version('transformers')
    return _transformers_version > version