from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
@classmethod
def get_default_description(cls):
    return f'[{cls.CODE}] {cls.MESSAGE}'