from __future__ import annotations
import copy
import json
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
def split_json(self, json_data: Dict[str, Any], convert_lists: bool=False) -> List[Dict]:
    """Splits JSON into a list of JSON chunks"""
    if convert_lists:
        chunks = self._json_split(self._list_to_dict_preprocessing(json_data))
    else:
        chunks = self._json_split(json_data)
    if not chunks[-1]:
        chunks.pop()
    return chunks