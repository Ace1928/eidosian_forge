import os
from functools import wraps
from typing import Any, Callable, Dict, TypeVar, cast
def requires(requirement: str) -> FuncT:
    """Decorate functions to gate features with wandb.require."""
    env_var = requirement_env_var_mapping[requirement]

    def deco(func: FuncT) -> FuncT:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not os.getenv(env_var):
                raise Exception(f'You need to enable this feature with `wandb.require({requirement!r})`')
            return func(*args, **kwargs)
        return cast(FuncT, wrapper)
    return cast(FuncT, deco)