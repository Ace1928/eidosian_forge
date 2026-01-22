import os
import stat
import sys
from io import BytesIO
from .. import errors, osutils, pyutils, tests
from .. import transport as _mod_transport
from .. import urlutils
from ..errors import ConnectionError, PathError, TransportNotPossible
from ..osutils import getcwd
from ..transport import (ConnectedTransport, FileExists, NoSuchFile, Transport,
from ..transport.memory import MemoryTransport
from ..transport.remote import RemoteTransport
from . import TestNotApplicable, TestSkipped, multiply_tests, test_server
from .test_transport import TestTransportImplementation
def transport_test_permutations():
    """Return a list of the klass, server_factory pairs to test."""
    result = []
    for module in _get_transport_modules():
        try:
            permutations = get_transport_test_permutations(pyutils.get_named_object(module))
            for klass, server_factory in permutations:
                scenario = ('{},{}'.format(klass.__name__, server_factory.__name__), {'transport_class': klass, 'transport_server': server_factory})
                result.append(scenario)
        except errors.DependencyNotPresent as e:
            pass
    return result