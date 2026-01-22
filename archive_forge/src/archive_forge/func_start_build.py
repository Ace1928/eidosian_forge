from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def start_build(self):
    result = None
    name = self.params.get('build_config_name')
    if not name:
        name = self.params.get('build_name')
    build_request = {'kind': 'BuildRequest', 'apiVersion': 'build.openshift.io/v1', 'metadata': {'name': name}, 'triggeredBy': [{'message': 'Manually triggered'}]}
    incremental = self.params.get('incremental')
    if incremental is not None:
        build_request.update({'sourceStrategyOptions': {'incremental': incremental}})
    if self.params.get('env_vars'):
        build_request.update({'env': self.params.get('env_vars')})
    if self.params.get('build_args'):
        build_request.update({'dockerStrategyOptions': {'buildArgs': self.params.get('build_args')}})
    no_cache = self.params.get('no_cache')
    if no_cache is not None:
        build_request.update({'dockerStrategyOptions': {'noCache': no_cache}})
    if self.params.get('commit'):
        build_request.update({'revision': {'git': {'commit': self.params.get('commit')}}})
    if self.params.get('build_config_name'):
        result = self.instantiate_build_config(name=self.params.get('build_config_name'), namespace=self.params.get('namespace'), request=build_request)
    else:
        result = self.clone_build(name=self.params.get('build_name'), namespace=self.params.get('namespace'), request=build_request)
    if result and self.params.get('wait'):
        start = datetime.now()

        def _total_wait_time():
            return (datetime.now() - start).seconds
        wait_timeout = self.params.get('wait_timeout')
        wait_sleep = self.params.get('wait_sleep')
        last_status_phase = None
        while _total_wait_time() < wait_timeout:
            params = dict(kind=result['kind'], api_version=result['apiVersion'], name=result['metadata']['name'], namespace=result['metadata']['namespace'])
            facts = self.kubernetes_facts(**params)
            if len(facts['resources']) > 0:
                last_status_phase = facts['resources'][0]['status']['phase']
                if last_status_phase == 'Complete':
                    result = facts['resources'][0]
                    break
                elif last_status_phase in ('Cancelled', 'Error', 'Failed'):
                    self.fail_json(msg='Unexpected status for Build %s/%s: %s' % (result['metadata']['name'], result['metadata']['namespace'], last_status_phase))
            time.sleep(wait_sleep)
        if last_status_phase != 'Complete':
            name = result['metadata']['name']
            namespace = result['metadata']['namespace']
            msg = 'Build %s/%s has not complete after %d second(s),current status is %s' % (namespace, name, wait_timeout, last_status_phase)
            self.fail_json(msg=msg)
    result = [result] if result else []
    self.exit_json(changed=True, builds=result)