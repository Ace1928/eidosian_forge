import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def deploy_template_show(self, identifier, fields=None, params=''):
    """Show specified baremetal deploy template.

        :param String identifier: Name or UUID of the deploy template
        :param List fields: List of fields to show
        :param List params: Additional kwargs
        :return: JSON object of deploy template
        """
    opts = self.get_opts(fields)
    output = self.openstack('baremetal deploy template show {0} {1} {2}'.format(opts, identifier, params))
    return json.loads(output)