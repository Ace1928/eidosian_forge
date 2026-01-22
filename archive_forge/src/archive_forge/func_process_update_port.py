import abc
from neutron_lib.api.definitions import portbindings
def process_update_port(self, plugin_context, data, result):
    """Process extended attributes for update port.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming port data
        :param result: port dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and update any extended port attributes defined by this
        driver. Extended attribute values, whether updated or not,
        must also be added to result.
        """
    pass