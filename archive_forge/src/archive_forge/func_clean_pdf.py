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
def clean_pdf(self, contents: str) -> str:
    """Clean the PDF file.

        Args:
            contents: a PDF file contents.

        Returns:

        """
    contents = '\n'.join([line for line in contents.split('\n') if not line.startswith('![]')])
    contents = contents.replace('\\section{', '# ').replace('}', '')
    contents = contents.replace('\\$', '$').replace('\\%', '%').replace('\\(', '(').replace('\\)', ')')
    return contents