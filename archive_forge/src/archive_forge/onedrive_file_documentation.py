from __future__ import annotations
import tempfile
from typing import TYPE_CHECKING, List
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
Load Documents