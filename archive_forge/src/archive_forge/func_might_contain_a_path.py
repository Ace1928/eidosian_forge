import os
from typing import Dict
def might_contain_a_path(candidate: str) -> bool:
    return os.sep in candidate