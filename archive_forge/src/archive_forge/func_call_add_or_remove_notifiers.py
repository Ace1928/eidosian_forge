from unittest import mock
from traits.observation._observe import add_or_remove_notifiers
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def call_add_or_remove_notifiers(**kwargs):
    """ Convenience function for calling add_or_remove_notifiers with default
    values.

    Parameters
    ----------
    **kwargs
        New argument values to use instead.
    """
    values = dict(object=mock.Mock(), graph=ObserverGraph(node=None), handler=mock.Mock(), target=_DEFAULT_TARGET, dispatcher=dispatch_same, remove=False)
    values.update(kwargs)
    add_or_remove_notifiers(**values)