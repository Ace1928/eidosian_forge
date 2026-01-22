import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def six_uuid5(namespace, name):
    """A uuid.uuid5 variant that takes utf-8 'name' both in Python 2 and 3.

    :param namespace: A UUID object used as a namespace in the generation of a
                      v5 UUID.
    :param name: Any string (either bytecode or unicode) used as a name in the
                 generation of a v5 UUID.
    :returns: A v5 UUID object.
    """
    return uuid.uuid5(namespace=namespace, name=name)