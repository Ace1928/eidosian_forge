from collections.abc import Mapping
from w3lib.http import headers_dict_to_raw
from scrapy.utils.datatypes import CaseInsensitiveDict, CaselessDict
from scrapy.utils.python import to_unicode
def normvalue(self, value):
    """Normalize values to bytes"""
    if value is None:
        value = []
    elif isinstance(value, (str, bytes)):
        value = [value]
    elif not hasattr(value, '__iter__'):
        value = [value]
    return [self._tobytes(x) for x in value]