import base64
from abc import ABC
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Literal, Optional, Union
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
def parse_issue(self, issue: dict) -> Document:
    """Create Document objects from a list of GitHub issues."""
    metadata = {'url': issue['html_url'], 'title': issue['title'], 'creator': issue['user']['login'], 'created_at': issue['created_at'], 'comments': issue['comments'], 'state': issue['state'], 'labels': [label['name'] for label in issue['labels']], 'assignee': issue['assignee']['login'] if issue['assignee'] else None, 'milestone': issue['milestone']['title'] if issue['milestone'] else None, 'locked': issue['locked'], 'number': issue['number'], 'is_pull_request': 'pull_request' in issue}
    content = issue['body'] if issue['body'] is not None else ''
    return Document(page_content=content, metadata=metadata)