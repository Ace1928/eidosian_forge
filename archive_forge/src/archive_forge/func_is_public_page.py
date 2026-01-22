import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def is_public_page(self, page: dict) -> bool:
    """Check if a page is publicly accessible."""
    restrictions = self.confluence.get_all_restrictions_for_content(page['id'])
    return page['status'] == 'current' and (not restrictions['read']['restrictions']['user']['results']) and (not restrictions['read']['restrictions']['group']['results'])