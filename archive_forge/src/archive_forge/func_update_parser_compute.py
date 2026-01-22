import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
def update_parser_compute(self, parser):
    parser.add_argument('--all-projects', action='store_true', default=False, help=self.enhance_help_nova_network(_('Display information from all projects (admin only)')))
    return parser