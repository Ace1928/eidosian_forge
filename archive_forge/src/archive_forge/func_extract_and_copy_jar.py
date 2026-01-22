import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from zipfile import ZipFile
from jinja2 import Environment, FileSystemLoader
from .. import run_command
def extract_and_copy_jar(wheel_path: Path, generated_files_path: Path) -> str:
    """
    extracts the PySide6 wheel and copies the 'jar' folder to 'generated_files_path'.
    These .jar files are added to the buildozer.spec file to be used later by buildozer
    """
    jar_path = generated_files_path / 'jar'
    jar_path.mkdir(parents=True, exist_ok=True)
    archive = ZipFile(wheel_path)
    jar_files = [file for file in archive.namelist() if file.startswith('PySide6/jar')]
    for file in jar_files:
        archive.extract(file, jar_path)
    return (jar_path / 'PySide6' / 'jar').resolve() if jar_files else None