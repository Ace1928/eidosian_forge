import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def create_filter(md_key: str, md_value: Any) -> str:
    md_filter_expr = self.config.field_name_mapping[md_key]
    if md_filter_expr is None:
        return ''
    expr = md_filter_expr.split(',')
    if len(expr) != 2:
        logger.error(f'filter {md_filter_expr} express is not correct, must contain mapping field and operator.')
        return ''
    md_filter_key = expr[0].strip()
    md_filter_operator = expr[1].strip()
    if isinstance(md_value, numbers.Number):
        return f'{md_filter_key} {md_filter_operator} {md_value}'
    return f'{md_filter_key}{md_filter_operator}"{md_value}"'