from jeepney.bus_messages import message_bus
from jeepney.io.blocking import DBusConnection, Proxy, open_dbus_connection
from secretstorage.collection import Collection, create_collection, \
from secretstorage.item import Item
from secretstorage.exceptions import SecretStorageException, \
from secretstorage.util import add_match_rules
def check_service_availability(connection: DBusConnection) -> bool:
    """Returns True if the Secret Service daemon is either running or
    available for activation via D-Bus, False otherwise.

    .. versionadded:: 3.2
    """
    from secretstorage.util import BUS_NAME
    proxy = Proxy(message_bus, connection)
    return proxy.NameHasOwner(BUS_NAME)[0] == 1 or BUS_NAME in proxy.ListActivatableNames()[0]