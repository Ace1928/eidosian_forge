import random
from . import testing
from .. import config
from .. import fixtures
from .. import util
from ..assertions import eq_
from ..assertions import is_false
from ..assertions import is_true
from ..config import requirements
from ..schema import Table
from ... import CheckConstraint
from ... import Column
from ... import ForeignKeyConstraint
from ... import Index
from ... import inspect
from ... import Integer
from ... import schema
from ... import String
from ... import UniqueConstraint
def ck(self, metadata, connection):
    convention = {'ck': 'check_constraint_%(table_name)s' + '_'.join((''.join((random.choice('abcdef') for j in range(30))) for i in range(10)))}
    metadata.naming_convention = convention
    cons = CheckConstraint('some_long_column_name > 5')
    Table('a_things_with_stuff', metadata, Column('id_long_column_name', Integer, primary_key=True), Column('some_long_column_name', Integer), cons)
    actual_name = cons.name
    metadata.create_all(connection)
    insp = inspect(connection)
    ck = insp.get_check_constraints('a_things_with_stuff')
    reflected_name = ck[0]['name']
    return (actual_name, reflected_name)