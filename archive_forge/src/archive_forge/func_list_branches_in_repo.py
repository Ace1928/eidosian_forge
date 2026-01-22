from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def list_branches_in_repo(self) -> str:
    """
        Fetches a list of all branches in the repository.

        Returns:
            str: A plaintext report containing the names of the branches.
        """
    try:
        branches = [branch.name for branch in self.github_repo_instance.get_branches()]
        if branches:
            branches_str = '\n'.join(branches)
            return f'Found {len(branches)} branches in the repository:\n{branches_str}'
        else:
            return 'No branches found in the repository'
    except Exception as e:
        return str(e)