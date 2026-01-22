import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def lazy_load_docs(self, query: str) -> Iterator[Document]:
    for d in self.lazy_load(query=query):
        yield self._dict2document(d)