import abc
from neutron_lib.api.definitions import portbindings
def update_network_precommit(self, context):
    """Update resources of a network.

        :param context: NetworkContext instance describing the new
            state of the network, as well as the original state prior
            to the update_network call.

        Update values of a network, updating the associated resources
        in the database. Called inside transaction context on session.
        Raising an exception will result in rollback of the
        transaction.

        update_network_precommit is called for all changes to the
        network state. It is up to the mechanism driver to ignore
        state or state changes that it does not know or care about.
        """
    pass