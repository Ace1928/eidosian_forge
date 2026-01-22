import functools
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
def get_value_absolute_path(self, full_value_name=False):
    path = []
    if self.value_name:
        if full_value_name:
            path.extend(self.translation_path[:-1])
        path.append(self.value_name)
    elif self.value_path:
        path.extend(self.value_path)
    if self.custom_value_path:
        path.extend(self.custom_value_path)
    return path