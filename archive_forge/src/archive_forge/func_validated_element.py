import io
import sys
import six
import types
from six import StringIO
from io import BytesIO
from lxml import etree
from ncclient import NCClientError
def validated_element(x, tags=None, attrs=None):
    """Checks if the root element of an XML document or Element meets the supplied criteria.

    *tags* if specified is either a single allowable tag name or sequence of allowable alternatives

    *attrs* if specified is a sequence of required attributes, each of which may be a sequence of several allowable alternatives

    Raises :exc:`XMLError` if the requirements are not met.
    """
    ele = to_ele(x)
    if tags:
        if isinstance(tags, (str, bytes)):
            tags = [tags]
        if ele.tag not in tags:
            raise XMLError('Element [%s] does not meet requirement' % ele.tag)
    if attrs:
        for req in attrs:
            if isinstance(req, (str, bytes)):
                req = [req]
            for alt in req:
                if alt in ele.attrib:
                    break
            else:
                raise XMLError('Element [%s] does not have required attributes' % ele.tag)
    return ele