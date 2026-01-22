import os
from typing import Any, List, Tuple
from jeepney import (
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import DBUS_UNKNOWN_METHOD, DBUS_NO_SUCH_OBJECT, \
from secretstorage.dhcrypto import Session, int_to_bytes
from secretstorage.exceptions import ItemNotFoundException, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
def exec_prompt(connection: DBusConnection, prompt_path: str) -> Tuple[bool, List[str]]:
    """Executes the prompt in a blocking mode.

    :returns: a tuple; the first element is a boolean value showing
              whether the operation was dismissed, the second element
              is a list of unlocked object paths
    """
    prompt = DBusAddressWrapper(prompt_path, PROMPT_IFACE, connection)
    rule = MatchRule(path=prompt_path, interface=PROMPT_IFACE, member='Completed', type=MessageType.signal)
    with connection.filter(rule) as signals:
        prompt.call('Prompt', 's', '')
        dismissed, result = connection.recv_until_filtered(signals).body
    assert dismissed is not None
    assert result is not None
    return (dismissed, result)