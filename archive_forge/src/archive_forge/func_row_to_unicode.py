import csv
import logging
import re
from io import StringIO
from typing import (
from warnings import warn
from lxml import etree
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.http import Response, TextResponse
from scrapy.selector import Selector
from scrapy.utils.python import re_rsearch, to_unicode
def row_to_unicode(row_: Iterable) -> List[str]:
    return [to_unicode(field, encoding) for field in row_]