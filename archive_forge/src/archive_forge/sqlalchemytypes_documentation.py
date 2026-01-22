import netaddr
from sqlalchemy import types
from neutron_lib._i18n import _
Truncates microseconds.

    Use this for datetime fields so we don't have to worry about DB-specific
    behavior when it comes to rounding/truncating microseconds off of
    timestamps.
    