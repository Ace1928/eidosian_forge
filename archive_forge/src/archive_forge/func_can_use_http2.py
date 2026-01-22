from __future__ import annotations
import sys
import openai
from .. import OpenAI, _load_client
from .._compat import model_json
from .._models import BaseModel
def can_use_http2() -> bool:
    try:
        import h2
    except ImportError:
        return False
    return True