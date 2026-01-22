from heat.common import template_format
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
Test that Template is parsable and publishes correct properties.