from oslo_utils import versionutils
from sqlalchemy import __version__
def driver_connection(connection):
    if sqla_2:
        return connection.connection.driver_connection
    else:
        return connection.connection.connection