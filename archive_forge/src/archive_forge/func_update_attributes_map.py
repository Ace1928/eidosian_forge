import abc
from neutron_lib._i18n import _
from neutron_lib import constants
@classmethod
def update_attributes_map(cls, extended_attributes, extension_attrs_map=None):
    """Update attributes map for this extension.

        Behaves like ExtensionDescriptor.update_attributes_map(), but
        if extension_attrs_map is not given the dict returned from
        self.get_extended_resources('2.0') is used.
        """
    if extension_attrs_map is None:
        extension_attrs_map = cls.get_extended_resources('2.0')
    super().update_attributes_map(extended_attributes, extension_attrs_map=extension_attrs_map)