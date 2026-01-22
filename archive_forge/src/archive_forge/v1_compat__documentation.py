from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v1 import docker_image as v1_image
from containerregistry.client.v2 import docker_digest
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util
from six.moves import zip  # pylint: disable=redefined-builtin
Override.