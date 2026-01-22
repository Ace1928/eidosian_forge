import logging
from typing import Callable, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.gcs_file import GCSFileLoader
from langchain_community.utilities.vertexai import get_client_info
Load documents.