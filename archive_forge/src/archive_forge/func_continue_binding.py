import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def continue_binding(self, segment_id, next_segments_to_bind):
    """Continue binding the port with different segments.

        :param segment_id: Network segment partially bound for the port.
        :param next_segments_to_bind: Segments to continue binding with.

        This method is called by MechanismDriver.bind_port to indicate
        it was able to partially bind the port, but that one or more
        additional mechanism drivers are required to complete the
        binding. The segment_id must identify an item in the current
        value of the segments_to_bind property. The list of segments
        IDs passed as next_segments_to_bind identify dynamic (or
        static) segments of the port's network that will be used to
        populate segments_to_bind for the next lower level of a
        hierarchical binding.
        """