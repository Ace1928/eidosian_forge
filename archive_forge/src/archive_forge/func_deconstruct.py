from typing import Any
from typing import Dict
from typing import Tuple
from django.db import models
from django.utils.translation import gettext_lazy as _
from . import ShortUUID
def deconstruct(self) -> Tuple[str, str, Tuple, Dict[str, Any]]:
    name, path, args, kwargs = super().deconstruct()
    kwargs['alphabet'] = self.alphabet
    kwargs['length'] = self.length
    kwargs['prefix'] = self.prefix
    kwargs.pop('default', None)
    return (name, path, args, kwargs)