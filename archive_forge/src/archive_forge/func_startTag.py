from __future__ import absolute_import, division, unicode_literals
from xml.dom import Node
from ..constants import namespaces, voidElements, spaceCharacters
def startTag(self, namespace, name, attrs):
    """Generates a StartTag token

        :arg namespace: the namespace of the token--can be ``None``

        :arg name: the name of the element

        :arg attrs: the attributes of the element as a dict

        :returns: StartTag token

        """
    return {'type': 'StartTag', 'name': name, 'namespace': namespace, 'data': attrs}