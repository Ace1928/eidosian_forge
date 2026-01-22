from functools import partial
from unittest import mock
import betamax
import fixtures
import requests
from keystoneauth1.fixture import hooks
from keystoneauth1.fixture import serializer as yaml_serializer
from keystoneauth1 import session
Determine the name of the selected serializer.

        If a class was specified, use the name attribute to generate this,
        otherwise, use the serializer_name parameter from ``__init__``.

        :returns:
            Name of the serializer
        :rtype:
            str
        