from __future__ import annotations
from typing import Iterator, List, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_community.document_loaders.base_o365 import (
from langchain_community.document_loaders.parsers.registry import get_parser
Load documents lazily. Use this when working at a large scale.