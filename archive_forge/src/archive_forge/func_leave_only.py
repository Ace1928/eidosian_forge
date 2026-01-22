import six
from ebooklib.plugins.base import BasePlugin
from ebooklib.utils import parse_html_string
def leave_only(item, tag_list):
    for _attr in six.iterkeys(item.attrib):
        if _attr not in tag_list:
            del item.attrib[_attr]