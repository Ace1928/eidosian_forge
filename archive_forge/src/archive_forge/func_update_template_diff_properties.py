import collections
import contextlib
import datetime as dt
import itertools
import pydoc
import re
import tenacity
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import short_id
from heat.common import timeutils
from heat.engine import attributes
from heat.engine.cfn import template as cfn_tmpl
from heat.engine import clients
from heat.engine.clients import default_client_plugin
from heat.engine import environment
from heat.engine import event
from heat.engine import function
from heat.engine.hot import template as hot_tmpl
from heat.engine import node_data
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import status
from heat.engine import support
from heat.engine import sync_point
from heat.engine import template
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_objects
from heat.objects import resource_properties_data as rpd_objects
from heat.rpc import client as rpc_client
def update_template_diff_properties(self, after_props, before_props):
    """Return changed Properties between the before and after properties.

        If any property having immutable as True is updated, raises
        NotSupported error.
        If any properties have changed which are not in
        update_allowed_properties, raises UpdateReplace.
        """
    update_allowed_set = self.calc_update_allowed(after_props)
    immutable_set = set()
    for psk, psv in after_props.props.items():
        if psv.immutable():
            immutable_set.add(psk)

    def prop_changed(key):
        try:
            before = before_props.get(key)
        except (TypeError, ValueError) as exc:
            LOG.warning('Ignoring error in old property value %(prop_name)s: %(msg)s', {'prop_name': key, 'msg': str(exc)})
            return True
        return before != after_props.get(key)
    changed_properties_set = set((k for k in after_props if prop_changed(k)))
    update_replace_forbidden = [k for k in changed_properties_set if k in immutable_set]
    if update_replace_forbidden:
        msg = _('Update to properties %(props)s of %(name)s (%(res)s)') % {'props': ', '.join(sorted(update_replace_forbidden)), 'res': self.type(), 'name': self.name}
        raise exception.NotSupported(feature=msg)
    if changed_properties_set and self.needs_replace_with_prop_diff(changed_properties_set, after_props, before_props):
        raise UpdateReplace(self)
    if not changed_properties_set.issubset(update_allowed_set):
        raise UpdateReplace(self.name)
    return dict(((k, after_props.get(k)) for k in changed_properties_set))