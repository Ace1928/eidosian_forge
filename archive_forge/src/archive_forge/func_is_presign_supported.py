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
def is_presign_supported(self) -> bool:
    config_endpoint = self.__endpoint + 'config'
    response = requests.get(config_endpoint, auth=self.__auth)
    response.raise_for_status()
    config = response.json()
    return config['storage_config']['pre_sign_support']