import sys
import typing as t
from types import CodeType
from types import TracebackType
from .exceptions import TemplateSyntaxError
from .utils import internal_code
from .utils import missing
def get_template_locals(real_locals: t.Mapping[str, t.Any]) -> t.Dict[str, t.Any]:
    """Based on the runtime locals, get the context that would be
    available at that point in the template.
    """
    ctx: 't.Optional[Context]' = real_locals.get('context')
    if ctx is not None:
        data: t.Dict[str, t.Any] = ctx.get_all().copy()
    else:
        data = {}
    local_overrides: t.Dict[str, t.Tuple[int, t.Any]] = {}
    for name, value in real_locals.items():
        if not name.startswith('l_') or value is missing:
            continue
        try:
            _, depth_str, name = name.split('_', 2)
            depth = int(depth_str)
        except ValueError:
            continue
        cur_depth = local_overrides.get(name, (-1,))[0]
        if cur_depth < depth:
            local_overrides[name] = (depth, value)
    for name, (_, value) in local_overrides.items():
        if value is missing:
            data.pop(name, None)
        else:
            data[name] = value
    return data