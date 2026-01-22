from tempfile import TemporaryFile
import threading
import pytest
from jeepney import (
from jeepney.io.threading import open_dbus_connection, DBusRouter, Proxy
@pytest.fixture()
def read_from_fd():
    name = 'io.gitlab.takluyver.jeepney.tests.read_from_fd'
    addr = DBusAddress(bus_name=name, object_path='/')
    with open_dbus_connection(bus='SESSION', enable_fds=True) as conn:
        with DBusRouter(conn) as router:
            status, = Proxy(message_bus, router).RequestName(name)
        assert status == 1

        def _reply_once():
            while True:
                msg = conn.receive()
                if msg.header.message_type is MessageType.method_call:
                    if msg.header.fields[HeaderFields.member] == 'ReadFD':
                        with msg.body[0].to_file('rb') as f:
                            f.seek(0)
                            b = f.read()
                        conn.send(new_method_return(msg, 'ay', (b,)))
                        return
                    else:
                        conn.send(new_error(msg, 'NoMethod'))
        reply_thread = threading.Thread(target=_reply_once, daemon=True)
        reply_thread.start()
        yield addr
    reply_thread.join()