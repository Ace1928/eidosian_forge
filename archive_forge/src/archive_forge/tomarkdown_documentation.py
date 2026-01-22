from __future__ import annotations
from typing import Iterator
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Lazily load the file.