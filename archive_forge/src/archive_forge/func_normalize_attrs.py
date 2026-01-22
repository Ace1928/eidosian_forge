import html.entities
import re
from .sgml import *
@staticmethod
def normalize_attrs(attrs):
    """
        :type attrs: List[Tuple[str, str]]
        :rtype: List[Tuple[str, str]]
        """
    if not attrs:
        return attrs
    attrs_d = {k.lower(): v for k, v in attrs}
    attrs = [(k, k in ('rel', 'type') and v.lower() or v) for k, v in attrs_d.items()]
    attrs.sort()
    return attrs