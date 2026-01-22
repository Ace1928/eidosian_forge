import json
import logging
import os
import uuid
from http import HTTPStatus
from typing import Any, Dict, Iterator, List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.pebblo import (
def get_source_size(self, source_path: str) -> int:
    """Fetch size of source path. Source can be a directory or a file.

        Args:
            source_path (str): Local path of data source.

        Returns:
            int: Source size in bytes.
        """
    if not source_path:
        return 0
    size = 0
    if os.path.isfile(source_path):
        size = os.path.getsize(source_path)
    elif os.path.isdir(source_path):
        total_size = 0
        for dirpath, _, filenames in os.walk(source_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        size = total_size
    return size