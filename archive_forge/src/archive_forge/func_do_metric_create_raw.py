import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('jsonbody', metavar='<JSON_BODY>', type=jsonutils.loads, help='The raw JSON body in single quotes. See api doc.')
def do_metric_create_raw(mc, args):
    """Create metric from raw json body."""
    try:
        mc.metrics.create(**args.jsonbody)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        print('Successfully created metric')