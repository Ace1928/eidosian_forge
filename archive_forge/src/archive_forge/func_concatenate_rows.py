import datetime
import json
from pathlib import Path
from typing import Iterator, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def concatenate_rows(row: dict) -> str:
    """Combine message information in a readable format ready to be used.

    Args:
        row: dictionary containing message information.
    """
    sender = row['sender_name']
    text = row['content']
    date = datetime.datetime.fromtimestamp(row['timestamp_ms'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    return f'{sender} on {date}: {text}\n\n'