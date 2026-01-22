from io import StringIO
from mimetypes import MimeTypes
from pkgutil import get_data
from typing import Dict, Mapping, Optional, Type, Union
from scrapy.http import Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import binary_is_text, to_bytes, to_unicode
def from_mimetype(self, mimetype: str) -> Type[Response]:
    """Return the most appropriate Response class for the given mimetype"""
    if mimetype is None:
        return Response
    if mimetype in self.classes:
        return self.classes[mimetype]
    basetype = f'{mimetype.split('/')[0]}/*'
    return self.classes.get(basetype, Response)