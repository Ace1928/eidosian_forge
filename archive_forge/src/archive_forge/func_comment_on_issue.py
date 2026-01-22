from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def comment_on_issue(self, comment_query: str) -> str:
    """
        Adds a comment to a github issue
        Parameters:
            comment_query(str): a string which contains the issue number,
            two newlines, and the comment.
            for example: "1

Working on it now"
            adds the comment "working on it now" to issue 1
        Returns:
            str: A success or failure message
        """
    issue_number = int(comment_query.split('\n\n')[0])
    comment = comment_query[len(str(issue_number)) + 2:]
    try:
        issue = self.github_repo_instance.get_issue(number=issue_number)
        issue.create_comment(comment)
        return 'Commented on issue ' + str(issue_number)
    except Exception as e:
        return 'Unable to make comment due to error:\n' + str(e)