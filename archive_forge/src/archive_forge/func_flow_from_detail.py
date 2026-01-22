import contextlib
from oslo_utils import importutils
from oslo_utils import reflection
import stevedore.driver
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence import backends as p_backends
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def flow_from_detail(flow_detail):
    """Reloads a flow previously saved.

    Gets the flow factories name and any arguments and keyword arguments from
    the flow details metadata, and then calls that factory to recreate the
    flow.

    :param flow_detail: FlowDetail that holds state of the flow to load
    """
    try:
        factory_data = flow_detail.meta['factory']
    except (KeyError, AttributeError, TypeError):
        raise ValueError('Cannot reconstruct flow %s %s: no factory information saved.' % (flow_detail.name, flow_detail.uuid))
    try:
        factory_fun = _fetch_factory(factory_data['name'])
    except (KeyError, ImportError):
        raise ImportError('Could not import factory for flow %s %s' % (flow_detail.name, flow_detail.uuid))
    args = factory_data.get('args', ())
    kwargs = factory_data.get('kwargs', {})
    return factory_fun(*args, **kwargs)