import os
import yaml
import typer
import contextlib
import subprocess
import concurrent.futures
from pathlib import Path
from pydantic import model_validator
from lazyops.types.models import BaseModel
from lazyops.libs.proxyobj import ProxyObject
from typing import Optional, List, Any, Dict, Union
def get_pip_requirements(kind: str, name: str) -> Optional[str]:
    """
    Helper for getting the pip requirements file for a given service/builder
    """
    pkg_dir = REQUIREMENTS_PATH.joinpath(kind)
    for alias in config.kinds[kind].get(name, []):
        req_file = pkg_dir.joinpath(f'{alias}.txt')
        if req_file.exists():
            req_file.write_text(req_file.read_text().replace('GITHUB_TOKEN', GITHUB_TOKEN))
            return req_file.as_posix()
        req_file = pkg_dir.joinpath(f'requirements.{alias}.txt')
        if req_file.exists():
            return req_file.as_posix()
    return None