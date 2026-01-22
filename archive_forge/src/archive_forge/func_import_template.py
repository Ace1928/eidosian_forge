from __future__ import (absolute_import, division, print_function)
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def import_template(module, connection):
    templates_service = connection.system_service().templates_service()
    if search_by_name(templates_service, module.params['name']) is not None:
        return False
    events_service = connection.system_service().events_service()
    last_event = events_service.list(max=1)[0]
    external_template = module.params['kvm']
    imports_service = connection.system_service().external_template_imports_service()
    imported_template = imports_service.add(otypes.ExternalTemplateImport(template=otypes.Template(name=module.params['name']), url=external_template.get('url'), cluster=otypes.Cluster(name=module.params['cluster']) if module.params['cluster'] else None, storage_domain=otypes.StorageDomain(name=external_template.get('storage_domain')) if external_template.get('storage_domain') else None, host=otypes.Host(name=external_template.get('host')) if external_template.get('host') else None, clone=external_template.get('clone', None)))
    templates_service = connection.system_service().templates_service()
    wait(service=templates_service.template_service(imported_template.template.id), condition=lambda tmp: len(events_service.list(from_=int(last_event.id), search='type=1158 and message=*%s*' % tmp.name)) > 0 if tmp is not None else False, fail_condition=lambda tmp: tmp is None, timeout=module.params['timeout'], poll_interval=module.params['poll_interval'])
    return True