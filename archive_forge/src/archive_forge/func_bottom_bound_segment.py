import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def bottom_bound_segment(self):
    """Return the current bottom-level bound segment dictionary.

        This property returns the current bottom-level bound segment
        dictionary, or None if the port is unbound. For a bound port,
        bottom_bound_segment is equivalent to
        binding_levels[-1][BOUND_SEGMENT], and returns the segment
        whose binding supplied the port's binding:vif_type and
        binding:vif_details attribute values.
        """