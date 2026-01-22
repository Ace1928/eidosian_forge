import logging
from oslo_vmware import vim_util
def get_portgroups(session, dvs_moref):
    """Gets all configured portgroups on the dvs_moref

    :param session: vCenter soap session
    :param dvs_moref: managed DVS object reference
    :returns: List of tuples that have the following format:
              (portgroup name, port group moref)
    """
    pgs = []
    port_groups = session.invoke_api(vim_util, 'get_object_properties', session.vim, dvs_moref, ['portgroup'])
    while port_groups:
        if len(port_groups) and hasattr(port_groups[0], 'propSet'):
            for prop in port_groups[0].propSet:
                for val in prop.val[0]:
                    props = session.invoke_api(vim_util, 'get_object_properties', session.vim, val, ['name'])
                    if len(props) and hasattr(props[0], 'propSet'):
                        for prop in props[0].propSet:
                            pgs.append((prop.val, val))
        port_groups = session._call_method(vim_util, 'continue_retrieval', port_groups)
    return pgs