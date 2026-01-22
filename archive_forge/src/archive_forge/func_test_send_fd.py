import pytest
from jeepney import new_method_call, MessageType, DBusAddress
from jeepney.bus_messages import message_bus, MatchRule
from jeepney.io.blocking import open_dbus_connection, Proxy
from .utils import have_session_bus
def test_send_fd(temp_file_and_contents, read_from_fd):
    temp_file, data = temp_file_and_contents
    readfd_call = new_method_call(read_from_fd, 'ReadFD', 'h', (temp_file,))
    with open_dbus_connection(bus='SESSION', enable_fds=True) as conn:
        reply = conn.send_and_get_reply(readfd_call, timeout=5)
    assert reply.header.message_type is MessageType.method_return
    assert reply.body[0] == data