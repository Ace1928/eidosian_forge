import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from zipfile import ZipFile
from jinja2 import Environment, FileSystemLoader
from .. import run_command
def find_qtlibs_in_wheel(wheel_pyside: Path):
    """
    Find the path to Qt/lib folder inside the wheel.
    """
    archive = ZipFile(wheel_pyside)
    qt_libs_path = wheel_pyside / 'PySide6/Qt/lib'
    qt_libs_path = zipfile.Path(archive, at=qt_libs_path)
    if not qt_libs_path.exists():
        for file in archive.namelist():
            if file.endswith('android-dependencies.xml'):
                qt_libs_path = zipfile.Path(archive, at=file).parent
                break
    if not qt_libs_path:
        raise FileNotFoundError('[DEPLOY] Unable to find Qt libs folder inside the wheel')
    return qt_libs_path