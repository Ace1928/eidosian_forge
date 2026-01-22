from lxml import etree
import sys
import re
import doctest
def temp_install(html=False, del_module=None):
    """
    Use this *inside* a doctest to enable this checker for this
    doctest only.

    If html is true, then by default the HTML parser will be used;
    otherwise the XML parser is used.
    """
    if html:
        Checker = LHTMLOutputChecker
    else:
        Checker = LXMLOutputChecker
    frame = _find_doctest_frame()
    dt_self = frame.f_locals['self']
    checker = Checker()
    old_checker = dt_self._checker
    dt_self._checker = checker
    check_func = frame.f_locals['check'].__func__
    checker_check_func = checker.check_output.__func__
    doctest.etree = etree
    _RestoreChecker(dt_self, old_checker, checker, check_func, checker_check_func, del_module)