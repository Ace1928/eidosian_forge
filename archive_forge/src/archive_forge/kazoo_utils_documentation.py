from kazoo import client
from kazoo import exceptions as k_exc
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow import exceptions as exc
from taskflow import logging
Creates a `kazoo`_ `client`_ given a configuration dictionary.

    :param conf: configuration dictionary that will be used to configure
                 the created client
    :type conf: dict

    The keys that will be extracted are:

    - ``read_only``: boolean that specifies whether to allow connections to
      read only servers, defaults to ``False``
    - ``randomize_hosts``: boolean that specifies whether to randomize
      host lists provided, defaults to ``False``
    - ``command_retry``: a kazoo `retry`_ object (or dict of options which
      will be used for creating one) that will be used for retrying commands
      that are executed
    - ``connection_retry``: a kazoo `retry`_ object (or dict of options which
      will be used for creating one)  that will be used for retrying
      connection failures that occur
    - ``hosts``: a string, list, set (or dict with host keys) that will
      specify the hosts the kazoo client should be connected to, if none
      is provided then ``localhost:2181`` will be used by default
    - ``timeout``: a float value that specifies the default timeout that the
      kazoo client will use
    - ``handler``: a kazoo handler object that can be used to provide the
      client with alternate async strategies (the default is `thread`_
      based, but `gevent`_, or `eventlet`_ ones can be provided as needed)
    - ``keyfile`` : SSL keyfile to use for authentication
    - ``keyfile_password``: SSL keyfile password
    -  ``certfile``: SSL certfile to use for authentication
    - ``ca``: SSL CA file to use for authentication
    - ``use_ssl``: argument to control whether SSL is used or not
    - ``verify_certs``: when using SSL, argument to bypass
        certs verification

    .. _client: https://kazoo.readthedocs.io/en/latest/api/client.html
    .. _kazoo: https://kazoo.readthedocs.io/
    .. _retry: https://kazoo.readthedocs.io/en/latest/api/retry.html
    .. _gevent: https://kazoo.readthedocs.io/en/latest/api/                handlers/gevent.html
    .. _eventlet: https://kazoo.readthedocs.io/en/latest/api/                  handlers/eventlet.html
    .. _thread: https://kazoo.readthedocs.io/en/latest/api/                handlers/threading.html
    