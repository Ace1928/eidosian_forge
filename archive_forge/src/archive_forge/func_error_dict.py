import json
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union
from .json import pydantic_encoder
from .utils import Representation
def error_dict(exc: Exception, config: Type['BaseConfig'], loc: 'Loc') -> 'ErrorDict':
    type_ = get_exc_type(exc.__class__)
    msg_template = config.error_msg_templates.get(type_) or getattr(exc, 'msg_template', None)
    ctx = exc.__dict__
    if msg_template:
        msg = msg_template.format(**ctx)
    else:
        msg = str(exc)
    d: 'ErrorDict' = {'loc': loc, 'msg': msg, 'type': type_}
    if ctx:
        d['ctx'] = ctx
    return d