from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def generate_substructure(self, scope, code):
    if not self.is_empty(scope):
        code.putln('')
        if self.ifdef:
            code.putln('#if %s' % self.ifdef)
        code.putln('static %s %s = {' % (self.slot_type, self.substructure_cname(scope)))
        for slot in self.sub_slots:
            slot.generate(scope, code)
        code.putln('};')
        if self.ifdef:
            code.putln('#endif')