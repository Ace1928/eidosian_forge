import sys
from pathlib import Path
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel
def get_validated_relative_path(root: Path, user_path: str) -> Path:
    """Resolve a relative path, raising an error if not within the root directory."""
    root = root.resolve()
    full_path = (root / user_path).resolve()
    if not is_relative_to(full_path, root):
        raise FileValidationError(f'Path {user_path} is outside of the allowed directory {root}')
    return full_path