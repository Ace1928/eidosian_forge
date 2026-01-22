from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
@pytest.mark.substrait
def test_expression_serialization_substrait():
    exprs = create_sample_expressions()
    schema = exprs['schema']
    for expr in exprs['literals']:
        serialized = expr.to_substrait(schema)
        deserialized = pc.Expression.from_substrait(serialized)
        assert expr.equals(deserialized)
    for expr in exprs['calls']:
        serialized = expr.to_substrait(schema)
        deserialized = pc.Expression.from_substrait(serialized)
        assert str(deserialized) == str(expr)
        serialized_again = deserialized.to_substrait(schema)
        deserialized_again = pc.Expression.from_substrait(serialized_again)
        assert deserialized.equals(deserialized_again)
    for expr, expr_norm in zip(exprs['refs'], exprs['numeric_refs']):
        serialized = expr.to_substrait(schema)
        deserialized = pc.Expression.from_substrait(serialized)
        assert str(deserialized) == str(expr_norm)
        serialized_again = deserialized.to_substrait(schema)
        deserialized_again = pc.Expression.from_substrait(serialized_again)
        assert deserialized.equals(deserialized_again)
    for expr in exprs['special']:
        serialized = expr.to_substrait(schema)
        deserialized = pc.Expression.from_substrait(serialized)
        serialized_again = deserialized.to_substrait(schema)
        deserialized_again = pc.Expression.from_substrait(serialized_again)
        assert deserialized.equals(deserialized_again)
    f = exprs['special'][0]
    serialized = f.to_substrait(schema)
    deserialized = pc.Expression.from_substrait(serialized)
    assert deserialized.equals(pc.scalar({'': 1}))
    a = pc.scalar(1)
    expr = a.isin([1, 2, 3])
    target = (a == 1) | (a == 2) | (a == 3)
    serialized = expr.to_substrait(schema)
    deserialized = pc.Expression.from_substrait(serialized)
    assert str(target) == str(deserialized)
    serialized_again = deserialized.to_substrait(schema)
    deserialized_again = pc.Expression.from_substrait(serialized_again)
    assert deserialized.equals(deserialized_again)