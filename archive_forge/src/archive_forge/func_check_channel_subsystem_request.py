import threading
from paramiko import util
from paramiko.common import (
def check_channel_subsystem_request(self, channel, name):
    """
        Determine if a requested subsystem will be provided to the client on
        the given channel.  If this method returns ``True``, all future I/O
        through this channel will be assumed to be connected to the requested
        subsystem.  An example of a subsystem is ``sftp``.

        The default implementation checks for a subsystem handler assigned via
        `.Transport.set_subsystem_handler`.
        If one has been set, the handler is invoked and this method returns
        ``True``.  Otherwise it returns ``False``.

        .. note:: Because the default implementation uses the `.Transport` to
            identify valid subsystems, you probably won't need to override this
            method.

        :param .Channel channel: the `.Channel` the pty request arrived on.
        :param str name: name of the requested subsystem.
        :return:
            ``True`` if this channel is now hooked up to the requested
            subsystem; ``False`` if that subsystem can't or won't be provided.
        """
    transport = channel.get_transport()
    handler_class, args, kwargs = transport._get_subsystem_handler(name)
    if handler_class is None:
        return False
    handler = handler_class(channel, name, self, *args, **kwargs)
    handler.start()
    return True