import base64
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Type
import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import Tool
def clear_files(self) -> None:
    self.files = {}