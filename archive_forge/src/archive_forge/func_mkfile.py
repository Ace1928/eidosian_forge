from pathlib import Path
import pytest
from ase.io import read
from ase.io.formats import UnknownFileTypeError
def mkfile(path, text):
    path = Path(path)
    path.write_text(text)
    return path