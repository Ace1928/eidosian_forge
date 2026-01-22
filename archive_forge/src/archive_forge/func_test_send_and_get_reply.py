import pytest
from jeepney import new_method_call, MessageType, DBusAddress
from jeepney.bus_messages import message_bus, MatchRule
from jeepney.io.blocking import open_dbus_connection, Proxy
from .utils import have_session_bus
def test_send_and_get_reply(session_conn):
    ping_call = new_method_call(bus_peer, 'Ping')
    reply = session_conn.send_and_get_reply(ping_call, timeout=5)
    assert reply.header.message_type == MessageType.method_return
    assert reply.body == ()
    ping_call = new_method_call(bus_peer, 'Ping')
    reply_body = session_conn.send_and_get_reply(ping_call, timeout=5, unwrap=True)
    assert reply_body == ()