import abc
from neutron_lib._i18n import _
from neutron_lib import constants
@classmethod
def get_optional_extensions(cls):
    """Returns the API definition's optional extensions."""
    cls._assert_api_definition('OPTIONAL_EXTENSIONS')
    return cls.api_definition.OPTIONAL_EXTENSIONS