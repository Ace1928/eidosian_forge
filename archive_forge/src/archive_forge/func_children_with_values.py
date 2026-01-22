import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def children_with_values(self):
    """Returns all children that has values

        :return: Possibly empty list of children.
        """
    childs = []
    for attribute in self._get_all_c_children_with_order():
        member = getattr(self, attribute)
        if member is None or member == []:
            pass
        elif isinstance(member, list):
            for instance in member:
                childs.append(instance)
        else:
            childs.append(member)
    return childs