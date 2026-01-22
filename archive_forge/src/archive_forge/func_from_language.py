from __future__ import annotations
import re
from typing import Any, List, Optional
from langchain_text_splitters.base import Language, TextSplitter
@classmethod
def from_language(cls, language: Language, **kwargs: Any) -> RecursiveCharacterTextSplitter:
    separators = cls.get_separators_for_language(language)
    return cls(separators=separators, is_separator_regex=True, **kwargs)