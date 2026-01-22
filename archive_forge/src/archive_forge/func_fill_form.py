from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def fill_form(el, values, form_id=None, form_index=None):
    el = _find_form(el, form_id=form_id, form_index=form_index)
    _fill_form(el, values)