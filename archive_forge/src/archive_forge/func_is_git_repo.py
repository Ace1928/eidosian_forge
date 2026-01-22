from __future__ import annotations
import os
import subprocess
def is_git_repo(dir: str) -> bool:
    """Is the given directory version-controlled with git?"""
    return os.path.exists(os.path.join(dir, '.git'))