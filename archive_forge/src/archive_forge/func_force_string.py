from pathlib import Path
from typing import Union, Dict, Any, List, Tuple
from collections import OrderedDict
def force_string(location):
    if isinstance(location, str):
        return location
    return str(location)