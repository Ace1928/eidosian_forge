from __future__ import absolute_import, division, print_function
from ansible.executor.task_executor import start_connection
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection as SocketConnection
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
Returns the path of the persistent connection socket.

        Attempts to ensure (within playcontext.timeout seconds) that the
        socket path exists. If the path exists (or the timeout has expired),
        returns the socket path.
        