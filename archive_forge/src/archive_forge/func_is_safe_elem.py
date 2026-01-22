import re
import six
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT
def is_safe_elem(self, tag, attrs):
    """Determine whether the given element should be considered safe for
        inclusion in the output.
        
        :param tag: the tag name of the element
        :type tag: QName
        :param attrs: the element attributes
        :type attrs: Attrs
        :return: whether the element should be considered safe
        :rtype: bool
        :since: version 0.6
        """
    if tag not in self.safe_tags:
        return False
    if tag.localname == 'input':
        input_type = attrs.get('type', '').lower()
        if input_type == 'password':
            return False
    return True