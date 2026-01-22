from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def replace_css(self, field_name, css, *processors, re=None, **kw):
    """
        Similar to :meth:`add_css` but replaces collected data instead of adding it.
        """
    values = self._get_cssvalues(css)
    self.replace_value(field_name, values, *processors, re=re, **kw)