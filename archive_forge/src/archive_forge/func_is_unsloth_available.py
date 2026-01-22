import importlib
import sys
def is_unsloth_available() -> bool:
    return importlib.util.find_spec('unsloth') is not None