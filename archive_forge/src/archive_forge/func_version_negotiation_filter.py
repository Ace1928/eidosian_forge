from heat.api.cfn import versions
from heat.api.middleware import version_negotiation as vn
def version_negotiation_filter(app, conf, **local_conf):
    return vn.VersionNegotiationFilter(versions.Controller, app, conf, **local_conf)