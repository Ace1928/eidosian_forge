from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def cleanup_output_data(self, data):
    if 'members' not in data['pool']:
        return []
    member_info = []
    for member in data['pool']['members']:
        member_info.append(member['id'])
    data['pool']['members'] = member_info