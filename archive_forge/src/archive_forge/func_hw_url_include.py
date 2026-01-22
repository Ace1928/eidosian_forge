from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def hw_url_include(self):
    if self.want.hw_url_include is None:
        return None
    if self.have.hw_url_include is None and self.want.hw_url_include == []:
        return None
    if self.have.hw_url_include is None:
        return self.want.hw_url_include
    wants = self.want.hw_url_include
    haves = list()
    for want in wants:
        for have in self.have.hw_url_include:
            if want['url'] == have['url']:
                entry = self._filter_have(want, have)
                haves.append(entry)
    result = compare_complex_list(wants, haves)
    return result