from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
A lazy loader for Documents.