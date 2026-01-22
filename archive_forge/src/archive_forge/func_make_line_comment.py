from typing import TYPE_CHECKING
from langchain_community.document_loaders.parsers.language.tree_sitter_segmenter import (  # noqa: E501
def make_line_comment(self, text: str) -> str:
    return f'// {text}'