import posixpath
from collections import defaultdict
from django.utils.safestring import mark_safe
from .base import Node, Template, TemplateSyntaxError, TextNode, Variable, token_kwargs
from .library import Library
def construct_relative_path(current_template_name, relative_name):
    """
    Convert a relative path (starting with './' or '../') to the full template
    name based on the current_template_name.
    """
    new_name = relative_name.strip('\'"')
    if not new_name.startswith(('./', '../')):
        return relative_name
    new_name = posixpath.normpath(posixpath.join(posixpath.dirname(current_template_name.lstrip('/')), new_name))
    if new_name.startswith('../'):
        raise TemplateSyntaxError("The relative path '%s' points outside the file hierarchy that template '%s' is in." % (relative_name, current_template_name))
    if current_template_name.lstrip('/') == new_name:
        raise TemplateSyntaxError("The relative path '%s' was translated to template name '%s', the same template in which the tag appears." % (relative_name, current_template_name))
    has_quotes = relative_name.startswith(('"', "'")) and relative_name[0] == relative_name[-1]
    return f'"{new_name}"' if has_quotes else new_name