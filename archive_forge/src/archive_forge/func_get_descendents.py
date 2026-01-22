import re
from urllib.parse import urljoin
import bs4
from bs4.element import Comment, NavigableString, Tag
def get_descendents(node):
    """An iterator over the all children tags (descendants) of this tag"""
    for desc in node.descendants:
        if isinstance(desc, bs4.Tag) and desc.name != 'template':
            yield desc