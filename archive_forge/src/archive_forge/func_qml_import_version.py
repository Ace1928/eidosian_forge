import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def qml_import_version(self) -> Tuple[int, int]:
    return (self._qml_import_major_version, self._qml_import_minor_version)