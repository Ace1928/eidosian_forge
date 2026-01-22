import warnings
from types import MappingProxyType
from typing import Any
from itemadapter._imports import attr, pydantic
def is_scrapy_item(obj: Any) -> bool:
    warnings.warn('itemadapter.utils.is_scrapy_item is deprecated and it will be removed in a future version', category=DeprecationWarning, stacklevel=2)
    from itemadapter.adapter import ScrapyItemAdapter
    return ScrapyItemAdapter.is_item(obj)