import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@project_access.setter
def project_access(self, value):
    self._project_access = value