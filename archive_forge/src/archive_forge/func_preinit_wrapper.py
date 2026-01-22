from typing import Any, Callable, Optional
import wandb
def preinit_wrapper(*args: Any, **kwargs: Any) -> Any:
    raise wandb.Error(f'You must call wandb.init() before {name}()')