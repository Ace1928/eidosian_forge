from __future__ import annotations
import base64
import hashlib
import json
import os
import tempfile
import uuid
from typing import Optional
import requests
def load_tiktoken_bpe(tiktoken_bpe_file: str, expected_hash: Optional[str]=None) -> dict[bytes, int]:
    contents = read_file_cached(tiktoken_bpe_file, expected_hash)
    return {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in contents.splitlines() if line)}