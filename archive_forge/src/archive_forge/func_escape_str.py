from __future__ import annotations
import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
def escape_str(self, value: str) -> str:
    """Escape special characters in a string for Clickhouse SQL queries.

        This method is used internally to prepare strings for safe insertion
        into SQL queries by escaping special characters that might otherwise
        interfere with the query syntax.

        Args:
            value: The string to be escaped.

        Returns:
            The escaped string, safe for insertion into SQL queries.
        """
    return ''.join((f'{self.BS}{c}' if c in self.must_escape else c for c in value))