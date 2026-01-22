from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
@classmethod
def override_attributes(cls, overrides: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:

    def attributes_transformer(element: Dict[str, Any]) -> Dict[str, Any]:
        element = cls.merge_dicts(element, overrides)
        return element
    return attributes_transformer