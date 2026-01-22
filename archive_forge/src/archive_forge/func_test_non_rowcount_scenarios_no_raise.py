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
@testing.variation('statement', ['update', 'delete', 'insert', 'select'])
@testing.variation('close_first', [True, False])
def test_non_rowcount_scenarios_no_raise(self, connection, statement, close_first):
    employees_table = self.tables.employees
    department = employees_table.c.department
    if statement.update:
        r = connection.execute(employees_table.update().where(department == 'C'), {'department': 'Z'})
    elif statement.delete:
        r = connection.execute(employees_table.delete().where(department == 'C'), {'department': 'Z'})
    elif statement.insert:
        r = connection.execute(employees_table.insert(), [{'employee_id': 25, 'name': 'none 1', 'department': 'X'}, {'employee_id': 26, 'name': 'none 2', 'department': 'Z'}, {'employee_id': 27, 'name': 'none 3', 'department': 'Z'}])
    elif statement.select:
        s = select(employees_table.c.name, employees_table.c.department).where(employees_table.c.department == 'C')
        r = connection.execute(s)
        r.all()
    else:
        statement.fail()
    if close_first:
        r.close()
    assert r.rowcount in (-1, 3)