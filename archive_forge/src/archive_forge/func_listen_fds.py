from socket import AF_UNSPEC as _AF_UNSPEC
from ._daemon import (__version__,
def listen_fds(unset_environment=True):
    """Return a list of socket activated descriptors

    Example::

      (in primary window)
      $ systemd-socket-activate -l 2000 python3 -c \\
          'from systemd.daemon import listen_fds; print(listen_fds())'
      (in another window)
      $ telnet localhost 2000
      (in primary window)
      ...
      Execing python3 (...)
      [3]
    """
    num = _listen_fds(unset_environment)
    return list(range(LISTEN_FDS_START, LISTEN_FDS_START + num))