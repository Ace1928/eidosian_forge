from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
def get_page(self, page: int) -> List[T]:
    params = dict(self.__firstParams)
    if page != 0:
        params['page'] = page + 1
    if self.__requester.per_page != 30:
        params['per_page'] = self.__requester.per_page
    headers, data = self.__requester.requestJsonAndCheck('GET', self.__firstUrl, parameters=params, headers=self.__headers)
    if self.__list_item in data:
        self.__totalCount = data.get('total_count')
        data = data[self.__list_item]
    return [self.__contentClass(self.__requester, headers, self._transformAttributes(element), completed=False) for element in data]