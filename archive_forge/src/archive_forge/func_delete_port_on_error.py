import collections
import contextlib
import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from webob import exc as web_exc
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib.api.definitions import network as net_apidef
from neutron_lib.api.definitions import port as port_apidef
from neutron_lib.api.definitions import portbindings as pb
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib.api.definitions import subnet as subnet_apidef
from neutron_lib import constants
from neutron_lib import exceptions
@contextlib.contextmanager
def delete_port_on_error(core_plugin, context, port_id):
    """A decorator that deletes a port upon exception.

    This decorator can be used to wrap a block of code that
    should delete a port if an exception is raised during the block's
    execution.

    :param core_plugin: The core plugin implementing the delete_port method to
        call.
    :param context: The context.
    :param port_id: The port's ID.
    :returns: None
    """
    try:
        yield
    except Exception:
        with excutils.save_and_reraise_exception():
            try:
                core_plugin.delete_port(context, port_id, l3_port_check=False)
            except exceptions.PortNotFound:
                LOG.debug('Port %s not found', port_id)
            except Exception:
                LOG.exception('Failed to delete port: %s', port_id)