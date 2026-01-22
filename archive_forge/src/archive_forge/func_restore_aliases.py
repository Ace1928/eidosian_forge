import inspect, os, sys, textwrap
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from traitlets import Bool
def restore_aliases(ip, alias=None):
    staliases = ip.db.get('stored_aliases', {})
    if alias is None:
        for k, v in staliases.items():
            ip.alias_manager.define_alias(k, v)
    else:
        ip.alias_manager.define_alias(alias, staliases[alias])