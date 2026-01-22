import logging
from oslo_utils import timeutils
from suds import sudsobject
def get_inventory_path(vim, entity_ref, max_objects=100):
    """Get the inventory path of a managed entity.

    :param vim: Vim object
    :param entity_ref: managed entity reference
    :param max_objects: maximum number of objects that should be returned in
                        a single call
    :return: inventory path of the entity_ref
    """
    client_factory = vim.client.factory
    property_collector = vim.service_content.propertyCollector
    prop_spec = build_property_spec(client_factory, 'ManagedEntity', ['name', 'parent'])
    select_set = build_selection_spec(client_factory, 'ParentTraversalSpec')
    select_set = build_traversal_spec(client_factory, 'ParentTraversalSpec', 'ManagedEntity', 'parent', False, [select_set])
    obj_spec = build_object_spec(client_factory, entity_ref, select_set)
    prop_filter_spec = build_property_filter_spec(client_factory, [prop_spec], [obj_spec])
    options = client_factory.create('ns0:RetrieveOptions')
    options.maxObjects = max_objects
    retrieve_result = vim.RetrievePropertiesEx(property_collector, specSet=[prop_filter_spec], options=options)
    entity_name = None
    propSet = None
    path = ''
    with WithRetrieval(vim, retrieve_result) as objects:
        for obj in objects:
            if hasattr(obj, 'propSet'):
                propSet = obj.propSet
                if len(propSet) >= 1 and (not entity_name):
                    entity_name = propSet[0].val
                elif len(propSet) >= 1:
                    path = '%s/%s' % (propSet[0].val, path)
    if propSet is not None and len(propSet) > 0:
        path = path[len(propSet[0].val):]
    if entity_name is None:
        entity_name = ''
    return '%s%s' % (path, entity_name)