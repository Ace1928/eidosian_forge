from decimal import Decimal
import uuid
from . import testing
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import Double
from ... import Float
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import Numeric
from ... import select
from ... import String
from ...types import LargeBinary
from ...types import UUID
from ...types import Uuid
@testing.combinations((Double(), 8.5514716, True), (Double(53), 8.5514716, True, testing.requires.float_or_double_precision_behaves_generically), (Float(), 8.5514, True), (Float(8), 8.5514, True, testing.requires.float_or_double_precision_behaves_generically), (Numeric(precision=15, scale=12, asdecimal=False), 8.5514716, True, testing.requires.literal_float_coercion), (Numeric(precision=15, scale=12, asdecimal=True), Decimal('8.5514716'), False), argnames='type_,value,do_rounding')
@testing.variation('sort_by_parameter_order', [True, False])
@testing.variation('multiple_rows', [True, False])
def test_insert_w_floats(self, connection, metadata, sort_by_parameter_order, type_, value, do_rounding, multiple_rows):
    """test #9701.

        this tests insertmanyvalues as well as decimal / floating point
        RETURNING types

        """
    t = Table('f_t', metadata, Column('id', Integer, Identity(), primary_key=True), Column('value', type_))
    t.create(connection)
    result = connection.execute(t.insert().returning(t.c.id, t.c.value, sort_by_parameter_order=bool(sort_by_parameter_order)), [{'value': value} for i in range(10)] if multiple_rows else {'value': value})
    if multiple_rows:
        i_range = range(1, 11)
    else:
        i_range = range(1, 2)
    if do_rounding:
        eq_({(id_, round(val_, 5)) for id_, val_ in result}, {(id_, round(value, 5)) for id_ in i_range})
        eq_({round(val_, 5) for val_ in connection.scalars(select(t.c.value))}, {round(value, 5)})
    else:
        eq_(set(result), {(id_, value) for id_ in i_range})
        eq_(set(connection.scalars(select(t.c.value))), {value})