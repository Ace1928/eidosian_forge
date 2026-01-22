import abc
from neutron_lib.api.definitions import portbindings
def create_subnet_precommit(self, context):
    """Allocate resources for a new subnet.

        :param context: SubnetContext instance describing the new
            subnet.

        Create a new subnet, allocating resources as necessary in the
        database. Called inside transaction context on session. Call
        cannot block.  Raising an exception will result in a rollback
        of the current transaction.
        """
    pass