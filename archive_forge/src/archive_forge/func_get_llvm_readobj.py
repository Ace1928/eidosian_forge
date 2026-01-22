import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from zipfile import ZipFile
from jinja2 import Environment, FileSystemLoader
from .. import run_command
def get_llvm_readobj(ndk_path: Path) -> Path:
    """
    Return the path to llvm_readobj from the Android Ndk
    """
    return ndk_path / 'toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readobj'