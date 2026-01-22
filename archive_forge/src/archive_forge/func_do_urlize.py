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
def do_urlize(eval_ctx: 'EvalContext', value: str, trim_url_limit: t.Optional[int]=None, nofollow: bool=False, target: t.Optional[str]=None, rel: t.Optional[str]=None, extra_schemes: t.Optional[t.Iterable[str]]=None) -> str:
    """Convert URLs in text into clickable links.

    This may not recognize links in some situations. Usually, a more
    comprehensive formatter, such as a Markdown library, is a better
    choice.

    Works on ``http://``, ``https://``, ``www.``, ``mailto:``, and email
    addresses. Links with trailing punctuation (periods, commas, closing
    parentheses) and leading punctuation (opening parentheses) are
    recognized excluding the punctuation. Email addresses that include
    header fields are not recognized (for example,
    ``mailto:address@example.com?cc=copy@example.com``).

    :param value: Original text containing URLs to link.
    :param trim_url_limit: Shorten displayed URL values to this length.
    :param nofollow: Add the ``rel=nofollow`` attribute to links.
    :param target: Add the ``target`` attribute to links.
    :param rel: Add the ``rel`` attribute to links.
    :param extra_schemes: Recognize URLs that start with these schemes
        in addition to the default behavior. Defaults to
        ``env.policies["urlize.extra_schemes"]``, which defaults to no
        extra schemes.

    .. versionchanged:: 3.0
        The ``extra_schemes`` parameter was added.

    .. versionchanged:: 3.0
        Generate ``https://`` links for URLs without a scheme.

    .. versionchanged:: 3.0
        The parsing rules were updated. Recognize email addresses with
        or without the ``mailto:`` scheme. Validate IP addresses. Ignore
        parentheses and brackets in more cases.

    .. versionchanged:: 2.8
       The ``target`` parameter was added.
    """
    policies = eval_ctx.environment.policies
    rel_parts = set((rel or '').split())
    if nofollow:
        rel_parts.add('nofollow')
    rel_parts.update((policies['urlize.rel'] or '').split())
    rel = ' '.join(sorted(rel_parts)) or None
    if target is None:
        target = policies['urlize.target']
    if extra_schemes is None:
        extra_schemes = policies['urlize.extra_schemes'] or ()
    for scheme in extra_schemes:
        if _uri_scheme_re.fullmatch(scheme) is None:
            raise FilterArgumentError(f'{scheme!r} is not a valid URI scheme prefix.')
    rv = urlize(value, trim_url_limit=trim_url_limit, rel=rel, target=target, extra_schemes=extra_schemes)
    if eval_ctx.autoescape:
        rv = Markup(rv)
    return rv