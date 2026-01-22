from __future__ import absolute_import, division, print_function
import re
import time
import xml.etree.ElementTree
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import iControlRestSession
from ..module_utils.teem import send_teem
def revoke_from_device(self):
    if self.module.check_mode:
        return True
    dossier = self.read_dossier_from_device()
    if dossier:
        self.want.update({'dossier': dossier})
    else:
        raise F5ModuleError('Dossier not generated.')
    if self.is_revoked():
        return False