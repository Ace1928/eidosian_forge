import re
from io import StringIO
import numba
from numba.core import types
from numba import jit, njit
from numba.tests.support import override_config, TestCase
import unittest
@TestCase.run_test_in_subprocess
def test_html_output_with_lifted_loop(self):
    """
        Test some format and behavior of the html annotation with lifted loop
        """

    @numba.jit(forceobj=True)
    def udt(x):
        object()
        z = 0
        for i in range(x):
            z += i
        return z
    re_lifted_tag = re.compile('<td class="lifted_tag">\\s*\\s*<details>\\s*<summary>\\s*<code>\\s*[0-9]+:\\s*[&nbsp;]+for i in range\\(x\\):  # this line is tagged\\s*', re.MULTILINE)
    sig_i64 = (types.int64,)
    udt.compile(sig_i64)
    cres = udt.overloads[sig_i64]
    buf = StringIO()
    cres.type_annotation.html_annotate(buf)
    output = buf.getvalue()
    buf.close()
    self.assertEqual(output.count('Function name: udt'), 1)
    sigfmt = 'with signature: {} -&gt; pyobject'
    self.assertEqual(output.count(sigfmt.format(sig_i64)), 1)
    self.assertEqual(len(re.findall(re_lifted_tag, output)), 1, msg='%s not found in %s' % (re_lifted_tag, output))
    sig_f64 = (types.float64,)
    udt.compile(sig_f64)
    cres = udt.overloads[sig_f64]
    buf = StringIO()
    cres.type_annotation.html_annotate(buf)
    output = buf.getvalue()
    buf.close()
    self.assertEqual(output.count('Function name: udt'), 2)
    self.assertEqual(output.count(sigfmt.format(sig_i64)), 1)
    self.assertEqual(output.count(sigfmt.format(sig_f64)), 1)
    self.assertEqual(len(re.findall(re_lifted_tag, output)), 2)