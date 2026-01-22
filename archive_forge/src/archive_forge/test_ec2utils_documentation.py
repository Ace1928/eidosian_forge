import testtools
from keystoneclient.contrib.ec2 import utils
from keystoneclient.tests.unit import client_fixtures
Test v4 generator with host:port format for malformed boto version.

        Validate for malformed version of boto, where the port should
        not be stripped.
        