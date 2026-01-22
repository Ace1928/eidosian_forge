from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def parse_issues(self, issues: List[Issue]) -> List[dict]:
    """
        Extracts title and number from each Issue and puts them in a dictionary
        Parameters:
            issues(List[Issue]): A list of Github Issue objects
        Returns:
            List[dict]: A dictionary of issue titles and numbers
        """
    parsed = []
    for issue in issues:
        title = issue.title
        number = issue.number
        opened_by = issue.user.login if issue.user else None
        issue_dict = {'title': title, 'number': number}
        if opened_by is not None:
            issue_dict['opened_by'] = opened_by
        parsed.append(issue_dict)
    return parsed