import re
import itertools
import textwrap
import uuid
import param
from param.display import register_display_accessor
from param._utils import async_executor
@depends(*resolve_ref(cb), watch=True)
def update_handle(*args, **kwargs):
    if handle is not None:
        handle.update(cb())