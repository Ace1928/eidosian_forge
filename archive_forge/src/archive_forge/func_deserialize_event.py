from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
from .constants import REPO_TYPE_MODEL
from .utils import parse_datetime
def deserialize_event(event: dict) -> DiscussionEvent:
    """Instantiates a [`DiscussionEvent`] from a dict"""
    event_id: str = event['id']
    event_type: str = event['type']
    created_at = parse_datetime(event['createdAt'])
    common_args = dict(id=event_id, type=event_type, created_at=created_at, author=event.get('author', {}).get('name', 'deleted'), _event=event)
    if event_type == 'comment':
        return DiscussionComment(**common_args, edited=event['data']['edited'], hidden=event['data']['hidden'], content=event['data']['latest']['raw'])
    if event_type == 'status-change':
        return DiscussionStatusChange(**common_args, new_status=event['data']['status'])
    if event_type == 'commit':
        return DiscussionCommit(**common_args, summary=event['data']['subject'], oid=event['data']['oid'])
    if event_type == 'title-change':
        return DiscussionTitleChange(**common_args, old_title=event['data']['from'], new_title=event['data']['to'])
    return DiscussionEvent(**common_args)