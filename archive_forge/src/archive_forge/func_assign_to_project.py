from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
def assign_to_project(self, project_name, urn):
    """Assign resource (urn) to project (name).

        Keyword arguments:
        project_name -- project name to associate the resource with
        urn -- resource URN (has the form do:resource_type:resource_id)

        Returns:
        assign_status -- ok, not_found, assigned, already_assigned, service_down
        error_message -- assignment error message (empty on success)
        resources -- resources assigned (or {} if error)

        Notes:
        For URN examples, see https://docs.digitalocean.com/reference/api/api-reference/#tag/Project-Resources

        Projects resources are identified by uniform resource names or URNs.
        A valid URN has the following format: do:resource_type:resource_id.

        The following resource types are supported:
        Resource Type  | Example URN
        Database       | do:dbaas:83c7a55f-0d84-4760-9245-aba076ec2fb2
        Domain         | do:domain:example.com
        Droplet        | do:droplet:4126873
        Floating IP    | do:floatingip:192.168.99.100
        Kubernetes     | do:kubernetes:bd5f5959-5e1e-4205-a714-a914373942af
        Load Balancer  | do:loadbalancer:39052d89-8dd4-4d49-8d5a-3c3b6b365b5b
        Space          | do:space:my-website-assets
        Volume         | do:volume:6fc4c277-ea5c-448a-93cd-dd496cfef71f
        """
    error_message, project = self.get_by_name(project_name)
    if not project:
        return ('', error_message, {})
    project_id = project.get('id', None)
    if not project_id:
        return ('', 'Unexpected error; cannot find project id for {0}'.format(project_name), {})
    data = {'resources': [urn]}
    response = self.rest.post('projects/{0}/resources'.format(project_id), data=data)
    status_code = response.status_code
    json = response.json
    if status_code != 200:
        message = json.get('message', 'No error message returned')
        return ('', 'Unable to assign resource {0} to project {1} [HTTP {2}: {3}]'.format(urn, project_name, status_code, message), {})
    resources = json.get('resources', [])
    if len(resources) == 0:
        return ('', 'Unexpected error; no resources returned (but assignment was successful)', {})
    if len(resources) > 1:
        return ('', 'Unexpected error; more than one resource returned (but assignment was successful)', {})
    status = resources[0].get('status', 'Unexpected error; no status returned (but assignment was successful)')
    return (status, 'Assigned {0} to project {1}'.format(urn, project_name), resources[0])