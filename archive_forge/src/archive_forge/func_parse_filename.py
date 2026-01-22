import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote
from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def parse_filename(self, url: str) -> str:
    """Parse the filename from an url.

        Args:
            url: Url to parse the filename from.

        Returns:
            The filename.
        """
    if (url_path := Path(url)) and url_path.suffix == '.pdf':
        return url_path.name
    else:
        return self._parse_filename_from_url(url)