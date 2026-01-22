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
def fk(self, metadata, connection):
    convention = {'fk': 'foreign_key_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s_' + '_'.join((''.join((random.choice('abcdef') for j in range(20))) for i in range(10)))}
    metadata.naming_convention = convention
    Table('a_things_with_stuff', metadata, Column('id_long_column_name', Integer, primary_key=True), test_needs_fk=True)
    cons = ForeignKeyConstraint(['aid'], ['a_things_with_stuff.id_long_column_name'])
    Table('b_related_things_of_value', metadata, Column('aid'), cons, test_needs_fk=True)
    actual_name = cons.name
    metadata.create_all(connection)
    if testing.requires.foreign_key_constraint_name_reflection.enabled:
        insp = inspect(connection)
        fks = insp.get_foreign_keys('b_related_things_of_value')
        reflected_name = fks[0]['name']
        return (actual_name, reflected_name)
    else:
        return (actual_name, None)