import logging
from typing import Any, Callable, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
Download a selected dataset.

        Returns: a list of Documents.

        