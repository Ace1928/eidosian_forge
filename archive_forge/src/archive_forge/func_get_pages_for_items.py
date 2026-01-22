import io
import mimetypes
from lxml import etree
def get_pages_for_items(items):
    pages_from_docs = [get_pages(item) for item in items]
    return [item for pages in pages_from_docs for item in pages]