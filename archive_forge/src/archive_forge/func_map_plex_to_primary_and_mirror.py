from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def map_plex_to_primary_and_mirror(self, plex_disks, disks, mirror_disks):
    """
        we have N plexes, and disks, and maybe mirror_disks
        we're trying to find which plex is used for disks, and which one, if applicable, for mirror_disks
        :return: a tuple with the names of the two plexes (disks_plex, mirror_disks_plex)
        the second one can be None
        """
    disks_plex = None
    mirror_disks_plex = None
    error = ''
    for plex in plex_disks:
        common = set(plex_disks[plex]).intersection(set(disks))
        if common:
            if disks_plex is None:
                disks_plex = plex
            else:
                error = 'found overlapping plexes: %s and %s' % (disks_plex, plex)
        if mirror_disks is not None:
            common = set(plex_disks[plex]).intersection(set(mirror_disks))
            if common:
                if mirror_disks_plex is None:
                    mirror_disks_plex = plex
                else:
                    error = 'found overlapping mirror plexes: %s and %s' % (mirror_disks_plex, plex)
    if not error:
        if disks_plex is None:
            error = 'cannot match disks with current aggregate disks'
        if mirror_disks is not None and mirror_disks_plex is None:
            if error:
                error += ', and '
            error += 'cannot match mirror_disks with current aggregate disks'
    if error:
        self.module.fail_json(msg='Error mapping disks for aggregate %s: %s.  Found: %s' % (self.parameters['name'], error, str(plex_disks)))
    return (disks_plex, mirror_disks_plex)