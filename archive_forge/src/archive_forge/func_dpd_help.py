from neutronclient._i18n import _
from neutronclient.common import exceptions
def dpd_help(policy):
    dpd = _(" %s Dead Peer Detection attributes. 'action'-hold,clear,disabled,restart,restart-by-peer. 'interval' and 'timeout' are non negative integers.  'interval' should be less than 'timeout' value.  'action', default:hold 'interval', default:30,  'timeout', default:120.") % policy.capitalize()
    return dpd