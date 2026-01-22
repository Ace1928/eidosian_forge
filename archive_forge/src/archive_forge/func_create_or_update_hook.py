from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def create_or_update_hook(self, project, hook_url, options):
    changed = False
    if self.hook_object is None:
        hook = self.create_hook(project, {'url': hook_url, 'push_events': options['push_events'], 'push_events_branch_filter': options['push_events_branch_filter'], 'issues_events': options['issues_events'], 'merge_requests_events': options['merge_requests_events'], 'tag_push_events': options['tag_push_events'], 'note_events': options['note_events'], 'job_events': options['job_events'], 'pipeline_events': options['pipeline_events'], 'wiki_page_events': options['wiki_page_events'], 'releases_events': options['releases_events'], 'enable_ssl_verification': options['enable_ssl_verification'], 'token': options['token']})
        changed = True
    else:
        changed, hook = self.update_hook(self.hook_object, {'push_events': options['push_events'], 'push_events_branch_filter': options['push_events_branch_filter'], 'issues_events': options['issues_events'], 'merge_requests_events': options['merge_requests_events'], 'tag_push_events': options['tag_push_events'], 'note_events': options['note_events'], 'job_events': options['job_events'], 'pipeline_events': options['pipeline_events'], 'wiki_page_events': options['wiki_page_events'], 'releases_events': options['releases_events'], 'enable_ssl_verification': options['enable_ssl_verification'], 'token': options['token']})
    self.hook_object = hook
    if changed:
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg='Successfully created or updated the hook %s' % hook_url)
        try:
            hook.save()
        except Exception as e:
            self._module.fail_json(msg='Failed to update hook: %s ' % e)
    return changed