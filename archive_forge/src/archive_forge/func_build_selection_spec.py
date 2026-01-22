import logging
from oslo_utils import timeutils
from suds import sudsobject
def build_selection_spec(client_factory, name):
    """Builds the selection spec.

    :param client_factory: factory to get API input specs
    :param name: name for the selection spec
    :returns: selection spec
    """
    sel_spec = client_factory.create('ns0:SelectionSpec')
    sel_spec.name = name
    return sel_spec