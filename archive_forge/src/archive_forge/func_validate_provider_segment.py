import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def validate_provider_segment(self, segment):
    """Validate attributes of a provider network segment.

        :param segment: segment dictionary using keys defined above
        :raises: neutron_lib.exceptions.InvalidInput if invalid

        Called outside transaction context to validate the provider
        attributes for a provider network segment. Raise InvalidInput
        if:

         - any required attribute is missing
         - any prohibited or unrecognized attribute is present
         - any attribute value is not valid

        The network_type attribute is present in segment, but
        need not be validated.
        """