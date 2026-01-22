import logging
from typing import Any, Dict, List, Mapping
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
def load_bibtex_entries(self, path: str) -> List[Dict[str, Any]]:
    """Load bibtex entries from the bibtex file at the given path."""
    import bibtexparser
    with open(path) as file:
        entries = bibtexparser.load(file).entries
    return entries