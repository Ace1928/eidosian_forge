import math
import random
import re
import typing
import typing as t
from collections import abc
from itertools import chain
from itertools import groupby
from markupsafe import escape
from markupsafe import Markup
from markupsafe import soft_str
from .async_utils import async_variant
from .async_utils import auto_aiter
from .async_utils import auto_await
from .async_utils import auto_to_list
from .exceptions import FilterArgumentError
from .runtime import Undefined
from .utils import htmlsafe_json_dumps
from .utils import pass_context
from .utils import pass_environment
from .utils import pass_eval_context
from .utils import pformat
from .utils import url_quote
from .utils import urlize
@pass_eval_context
def do_xmlattr(eval_ctx: 'EvalContext', d: t.Mapping[str, t.Any], autospace: bool=True) -> str:
    """Create an SGML/XML attribute string based on the items in a dict.

    If any key contains a space, this fails with a ``ValueError``. Values that
    are neither ``none`` nor ``undefined`` are automatically escaped.

    .. sourcecode:: html+jinja

        <ul{{ {'class': 'my_list', 'missing': none,
                'id': 'list-%d'|format(variable)}|xmlattr }}>
        ...
        </ul>

    Results in something like this:

    .. sourcecode:: html

        <ul class="my_list" id="list-42">
        ...
        </ul>

    As you can see it automatically prepends a space in front of the item
    if the filter returned something unless the second parameter is false.

    .. versionchanged:: 3.1.3
        Keys with spaces are not allowed.
    """
    items = []
    for key, value in d.items():
        if value is None or isinstance(value, Undefined):
            continue
        if _space_re.search(key) is not None:
            raise ValueError(f"Spaces are not allowed in attributes: '{key}'")
        items.append(f'{escape(key)}="{escape(value)}"')
    rv = ' '.join(items)
    if autospace and rv:
        rv = ' ' + rv
    if eval_ctx.autoescape:
        rv = Markup(rv)
    return rv