from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as mistral_client
from heat.engine.resources.openstack.vitrage.vitrage_template import \
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
wrong result format for vitrage templete validate