from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def fz_new_display_list_from_svg_xml_outparams_fn(xmldoc, xml, base_uri, dir):
    """
    Class-aware helper for out-params of fz_new_display_list_from_svg_xml() [fz_new_display_list_from_svg_xml()].
    """
    ret, w, h = ll_fz_new_display_list_from_svg_xml(xmldoc.m_internal, xml.m_internal, base_uri, dir.m_internal)
    return (FzDisplayList(ret), w, h)