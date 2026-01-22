from io import BytesIO
from io import StringIO
from lxml import etree
from bs4.element import (
from bs4.builder import (
from bs4.dammit import EncodingDetector
def parser_for(self, encoding):
    """Instantiate an appropriate parser for the given encoding.

        :param encoding: A string.
        :return: A parser object such as an `etree.XMLParser`.
        """
    parser = self.default_parser(encoding)
    if isinstance(parser, Callable):
        parser = parser(target=self, strip_cdata=False, recover=True, encoding=encoding)
    return parser