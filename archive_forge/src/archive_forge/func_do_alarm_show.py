import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('id', metavar='<ALARM_ID>', help='The ID of the alarm.')
def do_alarm_show(mc, args):
    """Describe the alarm."""
    fields = {}
    fields['alarm_id'] = args.id
    try:
        alarm = mc.alarms.get(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        if args.json:
            print(utils.json_formatter(alarm))
            return
        formatters = {'id': utils.json_formatter, 'alarm_definition': utils.json_formatter, 'metrics': utils.json_formatter, 'state': utils.json_formatter, 'links': utils.format_dictlist}
        utils.print_dict(alarm, formatters=formatters)