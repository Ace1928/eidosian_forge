from typing import Dict, Iterator, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX, SS_PATH
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, ItemNotFoundException, \
from secretstorage.item import Item
from secretstorage.util import DBusAddressWrapper, exec_prompt, \
def get_all_items(self) -> Iterator[Item]:
    """Returns a generator of all items in the collection."""
    for item_path in self._collection.get_property('Items'):
        yield Item(self.connection, item_path, self.session)