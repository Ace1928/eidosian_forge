from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
@classmethod
def get_resp_model(cls):
    if cls.__dict__.get('resp_model') is not None:
        return cls.resp_model
    cls.resp_model = cls.build_resp_model()
    return cls.resp_model