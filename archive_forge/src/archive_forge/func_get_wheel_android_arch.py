import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from zipfile import ZipFile
from jinja2 import Environment, FileSystemLoader
from .. import run_command
def get_wheel_android_arch(wheel: Path):
    """
    Get android architecture from wheel
    """
    supported_archs = ['aarch64', 'armv7a', 'i686', 'x86_64']
    for arch in supported_archs:
        if arch in wheel.stem:
            return arch
    return None