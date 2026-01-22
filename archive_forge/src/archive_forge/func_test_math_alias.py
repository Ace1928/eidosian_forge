import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('math_op, alias', [('truediv', 'divide'), ('truediv', 'div'), ('rtruediv', 'rdiv'), ('mul', 'multiply'), ('sub', 'subtract'), ('add', '__add__'), ('radd', '__radd__'), ('truediv', '__truediv__'), ('rtruediv', '__rtruediv__'), ('floordiv', '__floordiv__'), ('rfloordiv', '__rfloordiv__'), ('mod', '__mod__'), ('rmod', '__rmod__'), ('mul', '__mul__'), ('rmul', '__rmul__'), ('pow', '__pow__'), ('rpow', '__rpow__'), ('sub', '__sub__'), ('rsub', '__rsub__')])
def test_math_alias(math_op, alias):
    assert getattr(pd.DataFrame, math_op) == getattr(pd.DataFrame, alias)