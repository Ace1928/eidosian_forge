import abc
from neutron_lib.agent import extension
Handle a network update event.

        Called on network update.

        :param context: RPC context.
        :param data: dict of network data.
        