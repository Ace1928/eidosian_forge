import warnings
import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe
import itertools
import sys
from typing import cast, List, Optional, Any, Iterable, Union, Literal, IO
from typing import Type as TypingType
from typing import Dict as TypingDict
from .schema import core, channels, mixins, Undefined, UndefinedType, SCHEMA_URL
from .data import data_transformers
from ... import utils, expr
from ...expr import core as _expr_core
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes
from .compiler import vegalite_compilers
from ...utils._vegafusion_data import (
from ...utils.core import DataFrameLike
from ...utils.data import DataType
def open_editor(self, *, fullscreen: bool=False) -> None:
    """Opens the chart specification in the Vega chart editor using the default browser.

        Parameters
        ----------
        fullscreen : bool
            If True, editor will open chart in fullscreen mode. Default False
        """
    import webbrowser
    webbrowser.open(self.to_url(fullscreen=fullscreen))