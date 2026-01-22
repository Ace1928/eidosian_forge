import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote
from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def check_bs4(self) -> None:
    """Check if BeautifulSoup4 is installed.

        Raises:
            ImportError: If BeautifulSoup4 is not installed.
        """
    try:
        import bs4
    except ImportError:
        raise ImportError('BeautifulSoup4 is required for BlackboardLoader. Please install it with `pip install beautifulsoup4`.')