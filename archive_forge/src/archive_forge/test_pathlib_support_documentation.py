from pathlib import Path
import io
import pytest
from ase.build import molecule
from ase.io import read, write
from ase.utils import PurePath, convert_string_to_fd, reader, writer
Test reading/writing in ASE on pathlib objects