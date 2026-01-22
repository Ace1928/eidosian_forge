from .integrations import (
from .trainer_utils import (
from .utils import logging
@classmethod
def pip_install(cls):
    return f'`pip install {cls.pip_package or cls.name}`'