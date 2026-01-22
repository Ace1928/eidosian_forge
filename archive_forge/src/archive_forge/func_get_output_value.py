from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def get_output_value(self, field_name):
    """
        Return the collected values parsed using the output processor, for the
        given field. This method doesn't populate or modify the item at all.
        """
    proc = self.get_output_processor(field_name)
    proc = wrap_loader_context(proc, self.context)
    value = self._values.get(field_name, [])
    try:
        return proc(value)
    except Exception as e:
        raise ValueError("Error with output processor: field=%r value=%r error='%s: %s'" % (field_name, value, type(e).__name__, str(e))) from e