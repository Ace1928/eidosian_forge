from __future__ import absolute_import, division, print_function
import os
import re
import sys
import tempfile
import traceback
from contextlib import contextmanager
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
def cmd_pull(self):
    result = dict(changed=False, actions=[])
    if not self.check_mode:
        for service in self.project.get_services(self.services, include_deps=False):
            if 'image' not in service.options:
                continue
            self.log('Pulling image for service %s' % service.name)
            old_image_id = ''
            try:
                image = service.image()
                if image and image.get('Id'):
                    old_image_id = image['Id']
            except NoSuchImageError:
                pass
            except Exception as exc:
                self.client.fail('Error: service image lookup failed - %s' % to_native(exc))
            out_redir_name, err_redir_name = make_redirection_tempfiles()
            try:
                with stdout_redirector(out_redir_name):
                    with stderr_redirector(err_redir_name):
                        service.pull(ignore_pull_failures=False)
            except Exception as exc:
                fail_reason = get_failure_info(exc, out_redir_name, err_redir_name, msg_format='Error: pull failed with %s')
                self.client.fail(**fail_reason)
            else:
                cleanup_redirection_tempfiles(out_redir_name, err_redir_name)
            new_image_id = ''
            try:
                image = service.image()
                if image and image.get('Id'):
                    new_image_id = image['Id']
            except NoSuchImageError as exc:
                self.client.fail('Error: service image lookup failed after pull - %s' % to_native(exc))
            if new_image_id != old_image_id:
                result['changed'] = True
                result['actions'].append(dict(service=service.name, pulled_image=dict(name=service.image_name, id=new_image_id)))
    return result