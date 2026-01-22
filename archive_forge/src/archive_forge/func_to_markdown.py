from __future__ import annotations
from collections.abc import (
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import (
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import (
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import (
import pandas.core.indexes.base as ibase
from pandas.core.indexes.multi import maybe_droplevels
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime
import pandas.io.formats.format as fmt
from pandas.io.formats.info import (
import pandas.plotting
@doc(klass=_shared_doc_kwargs['klass'], storage_options=_shared_docs['storage_options'], examples=dedent('Examples\n            --------\n            >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")\n            >>> print(s.to_markdown())\n            |    | animal   |\n            |---:|:---------|\n            |  0 | elk      |\n            |  1 | pig      |\n            |  2 | dog      |\n            |  3 | quetzal  |\n\n            Output markdown with a tabulate option.\n\n            >>> print(s.to_markdown(tablefmt="grid"))\n            +----+----------+\n            |    | animal   |\n            +====+==========+\n            |  0 | elk      |\n            +----+----------+\n            |  1 | pig      |\n            +----+----------+\n            |  2 | dog      |\n            +----+----------+\n            |  3 | quetzal  |\n            +----+----------+'))
def to_markdown(self, buf: IO[str] | None=None, mode: str='wt', index: bool=True, storage_options: StorageOptions | None=None, **kwargs) -> str | None:
    """
        Print {klass} in Markdown-friendly format.

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.

        {storage_options}

        **kwargs
            These parameters will be passed to `tabulate                 <https://pypi.org/project/tabulate>`_.

        Returns
        -------
        str
            {klass} in Markdown-friendly format.

        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.

        {examples}
        """
    return self.to_frame().to_markdown(buf, mode=mode, index=index, storage_options=storage_options, **kwargs)