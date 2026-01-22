from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
@classmethod
def get_error_model(cls):
    if cls.__dict__.get('error_model') is not None:
        return cls.error_model
    cls.error_model = cls.build_error_model()
    return cls.error_model