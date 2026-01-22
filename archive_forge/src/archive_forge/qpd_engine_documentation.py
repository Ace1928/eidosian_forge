from abc import ABC, abstractmethod
from builtins import bool
from typing import Any, Dict, List, Tuple
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
`cols` must be in the format of
        `when1`, `value1`, ... ,`default`, and length must be >= 3.
        The reason to design the interface in this way is to simplify the translation
        from SQL to engine APIs.

        :return: [description]
        :rtype: Column
        