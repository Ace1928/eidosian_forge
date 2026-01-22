import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('id', metavar='<ALARM_ID>', help='The ID of the alarm.')
@utils.arg('state', metavar='<ALARM_STATE>', help='ALARM_STATE is one of [UNDETERMINED, OK, ALARM].')
@utils.arg('lifecycle_state', metavar='<LIFECYCLE_STATE>', help='The lifecycle state of the alarm.')
@utils.arg('link', metavar='<LINK>', help='A link to an external resource with information about the alarm.')
def do_alarm_update(mc, args):
    """Update the alarm state."""
    fields = {}
    fields['alarm_id'] = args.id
    if args.state.upper() not in state_types:
        errmsg = 'Invalid state, not one of [' + ', '.join(state_types) + ']'
        print(errmsg)
        return
    fields['state'] = args.state
    fields['lifecycle_state'] = args.lifecycle_state
    fields['link'] = args.link
    try:
        alarm = mc.alarms.update(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        print(jsonutils.dumps(alarm, indent=2))