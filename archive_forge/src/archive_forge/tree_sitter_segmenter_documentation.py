from abc import abstractmethod
from typing import TYPE_CHECKING, List
from langchain_community.document_loaders.parsers.language.code_segmenter import (
Abstract class for `CodeSegmenter`s that use the tree-sitter library.