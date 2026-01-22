import re
import shlex
from typing import List
from mlflow.utils.os import is_windows
def is_string_type(item):
    return isinstance(item, str)