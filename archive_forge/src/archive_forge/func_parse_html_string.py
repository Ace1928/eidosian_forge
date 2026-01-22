import io
import mimetypes
from lxml import etree
def parse_html_string(s):
    from lxml import html
    utf8_parser = html.HTMLParser(encoding='utf-8')
    html_tree = html.document_fromstring(s, parser=utf8_parser)
    return html_tree