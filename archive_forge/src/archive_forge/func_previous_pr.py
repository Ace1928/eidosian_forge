import json
import uuid
from typing import Optional
import requests
from huggingface_hub import Discussion, HfApi, get_repo_discussions
from .utils import cached_file, logging
def previous_pr(api: HfApi, model_id: str, pr_title: str, token: str) -> Optional['Discussion']:
    main_commit = api.list_repo_commits(model_id, token=token)[0].commit_id
    for discussion in get_repo_discussions(repo_id=model_id, token=token):
        if discussion.title == pr_title and discussion.status == 'open' and discussion.is_pull_request:
            commits = api.list_repo_commits(model_id, revision=discussion.git_reference, token=token)
            if main_commit == commits[1].commit_id:
                return discussion
    return None