from keystone.cmd.doctor import caching
from keystone.cmd.doctor import credential
from keystone.cmd.doctor import database
from keystone.cmd.doctor import debug
from keystone.cmd.doctor import federation
from keystone.cmd.doctor import ldap
from keystone.cmd.doctor import security_compliance
from keystone.cmd.doctor import tokens
from keystone.cmd.doctor import tokens_fernet
import keystone.conf
from keystone.i18n import _
def gather_symptoms():
    """Gather all of the objects in this module that are named symptom_*."""
    symptoms = []
    for module in SYMPTOM_MODULES:
        for name in dir(module):
            if name.startswith(SYMPTOM_PREFIX):
                symptoms.append(getattr(module, name))
    return symptoms