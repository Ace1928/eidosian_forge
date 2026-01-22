import abc
from neutron_lib.api.definitions import portbindings
def create_subnet_postcommit(self, context):
    """Create a subnet.

        :param context: SubnetContext instance describing the new
            subnet.

        Called after the transaction commits. Call can block, though
        will block the entire process so care should be taken to not
        drastically affect performance. Raising an exception will
        cause the deletion of the resource.
        """
    pass