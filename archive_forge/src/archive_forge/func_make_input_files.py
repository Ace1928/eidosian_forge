import base64
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Type
import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import Tool
def make_input_files(self) -> List[dict]:
    files = []
    for target_path, file_info in self.files.items():
        files.append({'pathname': target_path, 'contentsBasesixtyfour': file_to_base64(file_info.source_path)})
    return files