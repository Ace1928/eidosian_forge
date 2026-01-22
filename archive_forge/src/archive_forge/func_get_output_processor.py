from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def get_output_processor(self, field_name):
    proc = getattr(self, '%s_out' % field_name, None)
    if not proc:
        proc = self._get_item_field_attr(field_name, 'output_processor', self.default_output_processor)
    return unbound_method(proc)