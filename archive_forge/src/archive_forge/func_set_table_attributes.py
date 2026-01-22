from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
def set_table_attributes(self, attributes: str) -> Styler:
    """
        Set the table attributes added to the ``<table>`` HTML element.

        These are items in addition to automatic (by default) ``id`` attribute.

        Parameters
        ----------
        attributes : str

        Returns
        -------
        Styler

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_td_classes: Set the DataFrame of strings added to the ``class``
            attribute of ``<td>`` HTML elements.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4))
        >>> df.style.set_table_attributes('class="pure-table"')  # doctest: +SKIP
        # ... <table class="pure-table"> ...
        """
    self.table_attributes = attributes
    return self