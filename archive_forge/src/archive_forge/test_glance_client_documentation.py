from unittest import mock
import uuid
from glanceclient import exc
from heat.engine.clients import client_exception as exception
from heat.engine.clients.os import glance
from heat.tests import common
from heat.tests import utils
Tests the find_image_by_name_or_id function.