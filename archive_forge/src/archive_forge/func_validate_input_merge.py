from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def validate_input_merge(self, template_exists):
    """
        Validate input after getting all the parameters from DNAC.
        "If mandate like deviceTypes, softwareType and language "
        "already present in DNAC for a template."
        "It is not required to be provided in playbook, "
        "but if it is new creation error will be thrown to provide these fields.

        Parameters:
            template_exists (bool) - True if template exists, else False.

        Returns:
            None
        """
    template_params = self.want.get('template_params')
    language = template_params.get('language').upper()
    if language:
        if language not in self.accepted_languages:
            self.msg = 'Invalid value language {0} .Accepted language values are {1}'.format(self.accepted_languages, language)
            self.status = 'failed'
            return self
    else:
        template_params['language'] = 'JINJA'
    if not template_exists:
        if not template_params.get('deviceTypes') or not template_params.get('softwareType'):
            self.msg = 'DeviceTypes and SoftwareType are required arguments to create Templates'
            self.status = 'failed'
            return self
    self.msg = 'Input validated for merging'
    self.status = 'success'
    return self