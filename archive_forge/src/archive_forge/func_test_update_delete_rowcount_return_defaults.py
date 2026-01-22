from sqlalchemy import bindparam
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import testing
from sqlalchemy import text
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures
@testing.variation('implicit_returning', [True, False])
@testing.variation('dml', [('update', testing.requires.update_returning), ('delete', testing.requires.delete_returning)])
def test_update_delete_rowcount_return_defaults(self, connection, implicit_returning, dml):
    """note this test should succeed for all RETURNING backends
        as of 2.0.  In
        Idf28379f8705e403a3c6a937f6a798a042ef2540 we changed rowcount to use
        len(rows) when we have implicit returning

        """
    if implicit_returning:
        employees_table = self.tables.employees
    else:
        employees_table = Table('employees', MetaData(), Column('employee_id', Integer, autoincrement=False, primary_key=True), Column('name', String(50)), Column('department', String(1)), implicit_returning=False)
    department = employees_table.c.department
    if dml.update:
        stmt = employees_table.update().where(department == 'C').values(name=employees_table.c.department + 'Z').return_defaults()
    elif dml.delete:
        stmt = employees_table.delete().where(department == 'C').return_defaults()
    else:
        dml.fail()
    r = connection.execute(stmt)
    eq_(r.rowcount, 3)