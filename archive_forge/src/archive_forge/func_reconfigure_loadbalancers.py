import copy
from heat.common import exception
from heat.common.i18n import _
from heat.engine import scheduler
def reconfigure_loadbalancers(load_balancers, id_list):
    """Notify the LoadBalancer to reload its config.

    This must be done after activation (instance in ACTIVE state), otherwise
    the instances' IP addresses may not be available.
    """
    for lb in load_balancers:
        existing_defn = lb.frozen_definition()
        props = copy.copy(existing_defn.properties(lb.properties_schema, lb.context).data)
        if 'Instances' in lb.properties_schema:
            props['Instances'] = id_list
        else:
            raise exception.Error(_("Unsupported resource '%s' in LoadBalancerNames") % lb.name)
        lb_defn = existing_defn.freeze(properties=props)
        scheduler.TaskRunner(lb.update, lb_defn)()