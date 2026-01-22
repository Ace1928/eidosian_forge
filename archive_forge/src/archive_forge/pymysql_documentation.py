from .mysqldb import MySQLDialect_mysqldb
from ...util import langhelpers
determine if pymysql has deprecated, changed the default of,
        or removed the 'reconnect' argument of connection.ping().

        See #10492 and
        https://github.com/PyMySQL/mysqlclient/discussions/651#discussioncomment-7308971
        for background.

        