import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
('PUT', '/v2/metadefs/namespaces/Namespace1/objects/Object1', {},
        [('description', 'UPDATED_DESCRIPTION'),
        ('name', 'Object1'),
        ('properties', ...),
        ('required', [])])