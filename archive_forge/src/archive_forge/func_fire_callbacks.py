import logging
from typing import Callable, Generic, List
from typing_extensions import ParamSpec  # Python 3.10+
def fire_callbacks(self, *args: P.args, **kwargs: P.kwargs) -> None:
    for cb in self.callback_list:
        try:
            cb(*args, **kwargs)
        except Exception as e:
            logger.exception('Exception in callback for %s registered with CUDA trace', self.name)