from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def list_open_pull_requests(self) -> str:
    """
        Fetches all open PRs from the repo

        Returns:
            str: A plaintext report containing the number of PRs
            and each PR's title and number.
        """
    pull_requests = self.github_repo_instance.get_pulls(state='open')
    if pull_requests.totalCount > 0:
        parsed_prs = self.parse_pull_requests(pull_requests)
        parsed_prs_str = 'Found ' + str(len(parsed_prs)) + ' pull requests:\n' + str(parsed_prs)
        return parsed_prs_str
    else:
        return 'No open pull requests available'