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
def test_raw_sql_rowcount(self, connection):
    result = connection.exec_driver_sql("update employees set department='Z' where department='C'")
    eq_(result.rowcount, 3)