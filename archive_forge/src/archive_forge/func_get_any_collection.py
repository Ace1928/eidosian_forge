from typing import Dict, Iterator, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX, SS_PATH
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, ItemNotFoundException, \
from secretstorage.item import Item
from secretstorage.util import DBusAddressWrapper, exec_prompt, \
def get_any_collection(connection: DBusConnection) -> Collection:
    """Returns any collection, in the following order of preference:

    - The default collection;
    - The "session" collection (usually temporary);
    - The first collection in the collections list."""
    try:
        return Collection(connection)
    except ItemNotFoundException:
        pass
    try:
        return Collection(connection, SESSION_COLLECTION)
    except ItemNotFoundException:
        pass
    collections = list(get_all_collections(connection))
    if collections:
        return collections[0]
    else:
        raise ItemNotFoundException('No collections found.')