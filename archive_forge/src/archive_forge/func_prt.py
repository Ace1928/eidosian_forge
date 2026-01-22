from __future__ import annotations
import json
import logging
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def prt(self, msg: str) -> None:
    with open('/tmp/debugjaguar.log', 'a') as file:
        print(f'msg={msg}', file=file, flush=True)