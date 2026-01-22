from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
def errors_responses(errors: Sequence[Type[BaseError]]=None):
    responses = {'default': {}}
    if errors:
        cnt = 1
        for error_cls in errors:
            responses[error_cls.CODE] = {'model': error_cls.get_resp_model(), 'description': error_cls.get_description()}
            cnt += 1
    return responses