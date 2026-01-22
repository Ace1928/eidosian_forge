from __future__ import annotations
import pathlib
from io import BytesIO, StringIO
from typing import Any, Dict, List, Tuple, TypedDict
import requests
from langchain_core.documents import Document
def split_text_from_url(self, url: str) -> List[Document]:
    """Split HTML from web URL

        Args:
            url: web URL
        """
    r = requests.get(url)
    return self.split_text_from_file(BytesIO(r.content))