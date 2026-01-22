import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def extract_class_name(line: str) -> str:
    """Extract class name from class definition in the form of "class {CLASS_NAME}({Type}):"."""
    start_token = 'class '
    end_token = '('
    start, end = (line.find(start_token) + len(start_token), line.find(end_token))
    return line[start:end]