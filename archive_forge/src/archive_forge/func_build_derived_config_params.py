import copy
import uuid
from swiftclient import client as sc
from swiftclient import utils as swiftclient_utils
from urllib import parse as urlparse
from heatclient._i18n import _
from heatclient import exc
from heatclient.v1 import software_configs
def build_derived_config_params(action, source, name, input_values, server_id, signal_transport, signal_id=None):
    if isinstance(source, software_configs.SoftwareConfig):
        source = source.to_dict()
    input_values = input_values or {}
    inputs = copy.deepcopy(source.get('inputs')) or []
    for inp in inputs:
        input_key = inp['name']
        inp['value'] = input_values.pop(input_key, inp.get('default'))
    for inpk, inpv in input_values.items():
        inputs.append({'name': inpk, 'type': 'String', 'value': inpv})
    inputs.extend([{'name': 'deploy_server_id', 'description': _('ID of the server being deployed to'), 'type': 'String', 'value': server_id}, {'name': 'deploy_action', 'description': _('Name of the current action being deployed'), 'type': 'String', 'value': action}, {'name': 'deploy_signal_transport', 'description': _('How the server should signal to heat with the deployment output values.'), 'type': 'String', 'value': signal_transport}])
    if signal_transport == 'TEMP_URL_SIGNAL':
        inputs.append({'name': 'deploy_signal_id', 'description': _('ID of signal to use for signaling output values'), 'type': 'String', 'value': signal_id})
        inputs.append({'name': 'deploy_signal_verb', 'description': _('HTTP verb to use for signaling output values'), 'type': 'String', 'value': 'PUT'})
    elif signal_transport != 'NO_SIGNAL':
        raise exc.CommandError(_('Unsupported signal transport %s') % signal_transport)
    return {'group': source.get('group') or 'Heat::Ungrouped', 'config': source.get('config') or '', 'options': source.get('options') or {}, 'inputs': inputs, 'outputs': source.get('outputs') or [], 'name': name}