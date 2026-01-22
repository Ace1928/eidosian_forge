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
def test_update_rowcount1(self, connection):
    employees_table = self.tables.employees
    department = employees_table.c.department
    r = connection.execute(employees_table.update().where(department == 'C'), {'department': 'Z'})
    assert r.rowcount == 3