from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.six import raise_from
def fqdn_valid(name, min_labels=1, allow_underscores=False):
    """
    Example:
      - 'srv.example.com' is community.general.fqdn_valid
      - 'foo_bar.example.com' is community.general.fqdn_valid(allow_underscores=True)
    """
    if ANOTHER_LIBRARY_IMPORT_ERROR:
        raise_from(AnsibleError('Python package fqdn must be installed to use this test.'), ANOTHER_LIBRARY_IMPORT_ERROR)
    fobj = FQDN(name, min_labels=min_labels, allow_underscores=allow_underscores)
    return fobj.is_valid