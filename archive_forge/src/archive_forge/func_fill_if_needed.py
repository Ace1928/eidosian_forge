import asyncio
import os
from typing import Awaitable, Tuple, Type, TypeVar, Union
from typing import Dict as TypeDict
from typing import List as TypeList
from pathlib import Path
from traitlets.traitlets import Dict, Float, List, default
from nbclient.util import ensure_async
import re
from .notebook_renderer import NotebookRenderer
from .utils import ENV_VARIABLE
def fill_if_needed(self, delay: Union[float, None]=None, notebook_name: Union[str, None]=None, **kwargs) -> None:
    """Start kernels until the pool is full

                Args:
                    - delay (Union[float, None], optional): Delay time before
                    starting refill kernel. Defaults to None.
                    - notebook_name (Union[str, None], optional): Name of notebook to
                    create kernel pool.
                    Defaults to None.
                """
    delay = delay if delay is not None else self.fill_delay
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    default_config: dict = self.kernel_pools_config.get('default', {})
    notebook_config: dict = self.kernel_pools_config.get(notebook_name, default_config)
    kernel_env_variables: dict = notebook_config.get('kernel_env_variables', default_config.get('kernel_env_variables', {}))
    kernel_size: int = notebook_config.get('pool_size', default_config.get('pool_size', 1))
    pool = self._pools.get(notebook_name, [])
    self._pools[notebook_name] = pool
    if 'path' not in kwargs:
        kwargs['path'] = os.path.dirname(notebook_name) if notebook_name is not None else self.root_dir
    kernel_env = os.environ.copy()
    kernel_env_arg = kwargs.get('env', {})
    kernel_env.update(kernel_env_arg)
    for key in kernel_env_variables:
        if key not in kernel_env:
            kernel_env[key] = kernel_env_variables[key]
    kernel_env[ENV_VARIABLE.VOILA_BASE_URL] = self.parent.base_url
    kernel_env[ENV_VARIABLE.VOILA_SERVER_URL] = self.parent.server_url
    kernel_env[ENV_VARIABLE.VOILA_APP_PORT] = str(self.parent.port)
    kernel_env[ENV_VARIABLE.VOILA_PREHEAT] = 'True'
    kwargs['env'] = kernel_env
    heated = len(pool)

    def task_counter(tk):
        nonlocal heated
        heated += 1
        if heated == kernel_size:
            self.log.info('Kernel pool of %s is filled with %s kernel(s)', notebook_name, kernel_size)
    for _ in range(kernel_size - len(pool)):
        task = loop.create_task(wait_before(delay, self._initialize(notebook_name, None, **kwargs)))
        pool.append(task)
        task.add_done_callback(task_counter)