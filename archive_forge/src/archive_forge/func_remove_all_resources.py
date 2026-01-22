import abc
import collections
import copy
import functools
import hashlib
from stevedore import extension
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine import template_files
from heat.objects import raw_template as template_object
def remove_all_resources(self):
    """Remove all the resources from the template."""
    if self.RESOURCES in self.t:
        self.t.update({self.RESOURCES: {}})