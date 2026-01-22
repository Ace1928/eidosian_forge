from __future__ import annotations
import ast
import json
import os
from io import StringIO
from sys import version_info
from typing import IO, TYPE_CHECKING, Any, Callable, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_community.tools import BaseTool, Tool
from langchain_community.tools.e2b_data_analysis.unparse import Unparser
def remove_uploaded_file(self, uploaded_file: UploadedFile) -> None:
    """Remove uploaded file from the sandbox."""
    self.session.filesystem.remove(uploaded_file.remote_path)
    self._uploaded_files = [f for f in self._uploaded_files if f.remote_path != uploaded_file.remote_path]
    self.description = self.description + '\n' + self.uploaded_files_description