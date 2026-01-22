from __future__ import annotations
import json
import traceback
from functools import partial
import tornado
from .settings_utils import SchemaHandler
from .translation_utils import (

        Get installed language packs.

        If `locale` is equals to "default", the default locale will be used.

        Parameters
        ----------
        locale: str, optional
            If no locale is provided, it will list all the installed language packs.
            Default is `None`.
        