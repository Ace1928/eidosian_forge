import logging
from oslo_utils import timeutils
from suds import sudsobject
def build_traversal_spec(client_factory, name, type_, path, skip, select_set):
    """Builds the traversal spec.

    :param client_factory: factory to get API input specs
    :param name: name for the traversal spec
    :param type_: type of the managed object
    :param path: property path of the managed object
    :param skip: whether or not to filter the object identified by param path
    :param select_set: set of selection specs specifying additional objects
                       to filter
    :returns: traversal spec
    """
    traversal_spec = client_factory.create('ns0:TraversalSpec')
    traversal_spec.name = name
    traversal_spec.type = type_
    traversal_spec.path = path
    traversal_spec.skip = skip
    traversal_spec.selectSet = select_set
    return traversal_spec