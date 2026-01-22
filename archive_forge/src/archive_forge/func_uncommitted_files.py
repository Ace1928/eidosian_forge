from __future__ import annotations
import os
import re
from typing import Any
from streamlit import util
@property
def uncommitted_files(self):
    if not self.is_valid():
        return None
    return [item.a_path for item in self.repo.index.diff(None)]