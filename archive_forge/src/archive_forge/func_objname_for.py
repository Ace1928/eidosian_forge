from __future__ import annotations
import json
import os
import pathlib
import pickle
import re
import sys
import typing as T
from ..backend.ninjabackend import ninja_quote
from ..compilers.compilers import lang_suffixes
def objname_for(self, src: str) -> str:
    objname = self.target_data.source2object[src]
    assert isinstance(objname, str)
    return objname