import os
import tempfile
from typing import Callable, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.utilities.vertexai import get_client_info
def default_loader_func(file_path: str) -> BaseLoader:
    return UnstructuredFileLoader(file_path)