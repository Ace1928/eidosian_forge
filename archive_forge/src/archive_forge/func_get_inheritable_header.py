from typing import Optional, TypeVar
def get_inheritable_header(obj) -> Optional[str]:
    return getattr(obj, _INHERITABLE_HEADER, None)