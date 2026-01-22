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
def pk(self, metadata, connection):
    convention = {'pk': 'primary_key_%(table_name)s_%(column_0_N_name)s' + '_'.join((''.join((random.choice('abcdef') for j in range(30))) for i in range(10)))}
    metadata.naming_convention = convention
    a = Table('a_things_with_stuff', metadata, Column('id_long_column_name', Integer, primary_key=True), Column('id_another_long_name', Integer, primary_key=True))
    cons = a.primary_key
    actual_name = cons.name
    metadata.create_all(connection)
    insp = inspect(connection)
    pk = insp.get_pk_constraint('a_things_with_stuff')
    reflected_name = pk['name']
    return (actual_name, reflected_name)