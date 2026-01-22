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
@staticmethod
def get_file_owner_from_path(file_path: str) -> str:
    """Fetch owner of local file path.

        Args:
            file_path (str): Local file path.

        Returns:
            str: Name of owner.
        """
    try:
        import pwd
        file_owner_uid = os.stat(file_path).st_uid
        file_owner_name = pwd.getpwuid(file_owner_uid).pw_name
    except Exception:
        file_owner_name = 'unknown'
    return file_owner_name