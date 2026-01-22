import collections
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def get_elements_from_api(file_path: Union[str, List[str], Path, List[Path], None]=None, file: Union[IO, Sequence[IO], None]=None, api_url: str='https://api.unstructured.io/general/v0/general', api_key: str='', **unstructured_kwargs: Any) -> List:
    """Retrieve a list of elements from the `Unstructured API`."""
    if (is_list := isinstance(file_path, list)):
        file_path = [str(path) for path in file_path]
    if isinstance(file, collections.abc.Sequence) or is_list:
        from unstructured.partition.api import partition_multiple_via_api
        _doc_elements = partition_multiple_via_api(filenames=file_path, files=file, api_key=api_key, api_url=api_url, **unstructured_kwargs)
        elements = []
        for _elements in _doc_elements:
            elements.extend(_elements)
        return elements
    else:
        from unstructured.partition.api import partition_via_api
        return partition_via_api(filename=str(file_path) if file_path is not None else None, file=file, api_key=api_key, api_url=api_url, **unstructured_kwargs)