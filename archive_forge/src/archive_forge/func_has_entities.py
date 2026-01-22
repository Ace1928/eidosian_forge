import re
from html.entities import name2codepoint
from typing import Iterable, Match, AnyStr, Optional, Pattern, Tuple, Union
from urllib.parse import urljoin
from w3lib.util import to_unicode
from w3lib.url import safe_url_string
from w3lib._types import StrOrBytes
def has_entities(text: AnyStr, encoding: Optional[str]=None) -> bool:
    return bool(_ent_re.search(to_unicode(text, encoding)))