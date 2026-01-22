import os
import tempfile
import urllib.parse
from typing import Any, List, Optional
from urllib.parse import urljoin
import requests
from langchain_core.documents import Document
from requests.auth import HTTPBasicAuth
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader
Initialize UnstructuredLakeFSLoader.

        Args:

        :param lakefs_access_key:
        :param lakefs_secret_key:
        :param lakefs_endpoint:
        :param repo:
        :param ref:
        