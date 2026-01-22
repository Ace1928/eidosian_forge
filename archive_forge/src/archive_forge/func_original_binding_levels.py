import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def original_binding_levels(self):
    """Return dictionaries describing the original binding levels.

        This property returns a list of dictionaries describing each
        original binding level if the port was previously bound, or
        None if the port was unbound. The content is as described for
        the binding_levels property.

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """