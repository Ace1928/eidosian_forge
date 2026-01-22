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
@testing.requires.sane_multi_rowcount
def test_multi_update_rowcount(self, connection):
    employees_table = self.tables.employees
    stmt = employees_table.update().where(employees_table.c.name == bindparam('emp_name')).values(department='C')
    r = connection.execute(stmt, [{'emp_name': 'Bob'}, {'emp_name': 'Cynthia'}, {'emp_name': 'nonexistent'}])
    eq_(r.rowcount, 2)