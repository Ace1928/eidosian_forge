import re
import six
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT
def is_safe_css(self, propname, value):
    """Determine whether the given css property declaration is to be
        considered safe for inclusion in the output.
        
        :param propname: the CSS property name
        :param value: the value of the property
        :return: whether the property value should be considered safe
        :rtype: bool
        :since: version 0.6
        """
    if propname not in self.safe_css:
        return False
    if propname.startswith('margin') and '-' in value:
        return False
    return True