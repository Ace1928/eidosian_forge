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
def select_or_reject(context: 'Context', value: 't.Iterable[V]', args: t.Tuple, kwargs: t.Dict[str, t.Any], modfunc: t.Callable[[t.Any], t.Any], lookup_attr: bool) -> 't.Iterator[V]':
    if value:
        func = prepare_select_or_reject(context, args, kwargs, modfunc, lookup_attr)
        for item in value:
            if func(item):
                yield item