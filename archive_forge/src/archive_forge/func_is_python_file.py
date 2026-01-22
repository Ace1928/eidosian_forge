import json
import os
import subprocess
import sys
from typing import List, Tuple
from pathlib import Path
from . import (METATYPES_JSON_SUFFIX, PROJECT_FILE_SUFFIX, qt_metatype_json_dir,
def is_python_file(file: Path) -> bool:
    return file.suffix == '.py' or (sys.platform == 'win32' and file.suffix == '.pyw')