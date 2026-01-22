import json
import logging
import os
import re
import tempfile
import time
from abc import ABC
from io import StringIO
from pathlib import Path
from typing import (
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
def send_pdf(self) -> str:
    with open(self.file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(self.url, headers=self._mathpix_headers, files=files, data=self.data)
    response_data = response.json()
    if 'error' in response_data:
        raise ValueError(f'Mathpix request failed: {response_data['error']}')
    if 'pdf_id' in response_data:
        pdf_id = response_data['pdf_id']
        return pdf_id
    else:
        raise ValueError('Unable to send PDF to Mathpix.')