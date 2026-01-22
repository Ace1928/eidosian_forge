import logging
from oslo_utils import timeutils
from suds import sudsobject
def get_properties_for_a_collection_of_objects(vim, type_, obj_list, properties, max_objects=None):
    """Gets the list of properties for the collection of
    objects of the type specified.
    """
    client_factory = vim.client.factory
    if len(obj_list) == 0:
        return []
    prop_spec = get_prop_spec(client_factory, type_, properties)
    lst_obj_specs = []
    for obj in obj_list:
        lst_obj_specs.append(get_obj_spec(client_factory, obj))
    prop_filter_spec = get_prop_filter_spec(client_factory, lst_obj_specs, [prop_spec])
    options = client_factory.create('ns0:RetrieveOptions')
    options.maxObjects = max_objects if max_objects else len(obj_list)
    return vim.RetrievePropertiesEx(vim.service_content.propertyCollector, specSet=[prop_filter_spec], options=options)