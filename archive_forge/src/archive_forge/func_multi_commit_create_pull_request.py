import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Union
from ._commit_api import CommitOperationAdd, CommitOperationDelete
from .community import DiscussionWithDetails
from .utils import experimental
from .utils._cache_manager import _format_size
from .utils.insecure_hashlib import sha256
def multi_commit_create_pull_request(api: 'HfApi', repo_id: str, commit_message: str, commit_description: Optional[str], strategy: MultiCommitStrategy, token: Optional[str], repo_type: Optional[str]) -> DiscussionWithDetails:
    return api.create_pull_request(repo_id=repo_id, title=f'[WIP] {commit_message} (multi-commit {strategy.id})', description=multi_commit_generate_comment(commit_message=commit_message, commit_description=commit_description, strategy=strategy), token=token, repo_type=repo_type)