import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
def class_factory(cls_name, attrs):
    """
    This class takes a name and a dictionary to create a class.
    The clkass has an init method, an iter for retrieving properties,
    and, finally, a repr for returning the instance
    :param cls_name: The name to be tacked onto the suffix NttCis
    :type cls_name: ``str``
    :param attrs: The attributes and values for an instance
    :type attrs: ``dict``
    :return:  a class that inherits from ClassFactory
    :rtype: ``ClassFactory``
    """

    def __init__(self, *args, **kwargs):
        for key in attrs:
            setattr(self, key, attrs[key])
        if cls_name == 'NttCisServer':
            self.state = self._get_state()

    def __iter__(self):
        for name in self.__dict__:
            yield getattr(self, name)

    def __repr__(self):
        values = ', '.join(('{}={!r}'.format(*i) for i in zip(self.__dict__, self)))
        return '{}({})'.format(self.__class__.__name__, values)
    cls_attrs = dict(__init__=__init__, __iter__=__iter__, __repr__=__repr__)
    return type('NttCis{}'.format(cls_name), (ClassFactory,), cls_attrs)