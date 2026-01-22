import os
from enum import Enum
from typing import Any
@classmethod
def from_module_name(cls, module_name: str) -> 'AppEnv':
    """
        Retrieves the app environment
        """
    module_name = module_name.replace('.', '_').upper()
    for key in {'SERVER_ENV', f'{module_name}_ENV', 'APP_ENV', 'ENVIRONMENT'}:
        if (env_value := os.getenv(key)):
            return cls.from_env(env_value)
    from lazyops.utils.system import is_in_kubernetes, get_host_name
    if is_in_kubernetes():
        hn = get_host_name()
        try:
            parts = hn.split('-')
            return cls.from_env(parts[2]) if len(parts) > 3 else cls.PRODUCTION
        except Exception as e:
            return cls.from_hostname(hn)
    return cls.LOCAL