from .mysqldb import MySQLDialect_mysqldb
from ...util import langhelpers
@langhelpers.memoized_property
def supports_server_side_cursors(self):
    try:
        cursors = __import__('pymysql.cursors').cursors
        self._sscursor = cursors.SSCursor
        return True
    except (ImportError, AttributeError):
        return False