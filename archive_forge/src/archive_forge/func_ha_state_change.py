import abc
from neutron_lib.agent import extension
@abc.abstractmethod
def ha_state_change(self, context, data):
    """Change router state from agent extension.

        Called on HA router state change.

        :param context: rpc context
        :param data: dict of router_id and new state
        """