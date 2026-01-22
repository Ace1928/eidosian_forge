from __future__ import annotations
import copy
import json
from .nbbase import from_dict
from .rwbase import NotebookReader, NotebookWriter, rejoin_lines, restore_bytes, split_lines
Convert a notebook object to a string.