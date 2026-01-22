import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_c_preprocessor(self, feature_name, tabs=0):
    """
        Generate C preprocessor definitions and include headers of a CPU feature.

        Parameters
        ----------
        'feature_name': str
            CPU feature name in uppercase.
        'tabs': int
            if > 0, align the generated strings to the right depend on number of tabs.

        Returns
        -------
        str, generated C preprocessor

        Examples
        --------
        >>> self.feature_c_preprocessor("SSE3")
        /** SSE3 **/
        #define NPY_HAVE_SSE3 1
        #include <pmmintrin.h>
        """
    assert feature_name.isupper()
    feature = self.feature_supported.get(feature_name)
    assert feature is not None
    prepr = ['/** %s **/' % feature_name, '#define %sHAVE_%s 1' % (self.conf_c_prefix, feature_name)]
    prepr += ['#include <%s>' % h for h in feature.get('headers', [])]
    extra_defs = feature.get('group', [])
    extra_defs += self.feature_extra_checks(feature_name)
    for edef in extra_defs:
        prepr += ['#ifndef %sHAVE_%s' % (self.conf_c_prefix, edef), '\t#define %sHAVE_%s 1' % (self.conf_c_prefix, edef), '#endif']
    if tabs > 0:
        prepr = ['\t' * tabs + l for l in prepr]
    return '\n'.join(prepr)