import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def release_dynamic_segment(self, segment_id):
    """Release an allocated dynamic segment.

        :param segment_id: UUID of the dynamic network segment.

        Called by the MechanismDriver.delete_port or update_port to release
        the dynamic segment allocated for this port.
        """