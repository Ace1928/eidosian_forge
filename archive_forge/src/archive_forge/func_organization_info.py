from __future__ import annotations
import sys
import openai
from .. import OpenAI, _load_client
from .._compat import model_json
from .._models import BaseModel
def organization_info() -> str:
    organization = openai.organization
    if organization is not None:
        return '[organization={}] '.format(organization)
    return ''