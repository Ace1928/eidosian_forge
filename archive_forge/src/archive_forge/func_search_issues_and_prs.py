from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def search_issues_and_prs(self, query: str) -> str:
    """
        Searches issues and pull requests in the repository.

        Parameters:
            query(str): The search query

        Returns:
            str: A string containing the first 5 issues and pull requests
        """
    search_result = self.github.search_issues(query, repo=self.github_repository)
    max_items = min(5, search_result.totalCount)
    results = [f'Top {max_items} results:']
    for issue in search_result[:max_items]:
        results.append(f'Title: {issue.title}, Number: {issue.number}, State: {issue.state}')
    return '\n'.join(results)