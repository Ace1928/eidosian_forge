import itertools
from oslo_log import log
import passlib.hash
import keystone.conf
from keystone import exception
from keystone.i18n import _
Hash a password. Harder.