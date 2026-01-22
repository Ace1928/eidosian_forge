from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Lazily load table schemas as Document objects.

        Yields:
            Document objects, each representing the schema of a table.
        