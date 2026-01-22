import logging
from oslo_utils import timeutils
from suds import sudsobject
def get_object_property(vim, moref, property_name, skip_op_id=False):
    """Get property of the given managed object.

    :param vim: Vim object
    :param moref: managed object reference
    :param property_name: name of the property to be retrieved
    :param skip_op_id: whether to skip putting opID in the request
    :returns: property of the given managed object
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException
    """
    props = get_object_properties(vim, moref, [property_name], skip_op_id=skip_op_id)
    prop_val = None
    if props:
        prop = None
        if hasattr(props[0], 'propSet'):
            prop = props[0].propSet
        if prop:
            prop_val = prop[0].val
    return prop_val