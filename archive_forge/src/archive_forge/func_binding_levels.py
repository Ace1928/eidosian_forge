import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def binding_levels(self):
    """Return dictionaries describing the current binding levels.

        This property returns a list of dictionaries describing each
        binding level if the port is bound or partially bound, or None
        if the port is unbound. Each returned dictionary contains the
        name of the bound driver under the BOUND_DRIVER key, and the
        bound segment dictionary under the BOUND_SEGMENT key.

        The first entry (index 0) describes the top-level binding,
        which always involves one of the port's network's static
        segments. In the case of a hierarchical binding, subsequent
        entries describe the lower-level bindings in descending order,
        which may involve dynamic segments. Adjacent levels where
        different drivers bind the same static or dynamic segment are
        possible. The last entry (index -1) describes the bottom-level
        binding that supplied the port's binding:vif_type and
        binding:vif_details attribute values.

        Within calls to MechanismDriver.bind_port, descriptions of the
        levels above the level currently being bound are returned.
        """