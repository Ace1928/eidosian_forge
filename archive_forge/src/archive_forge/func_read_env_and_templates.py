from openstack import exceptions
from openstack.orchestration.util import template_utils
from openstack.orchestration.v1 import resource as _resource
from openstack.orchestration.v1 import software_config as _sc
from openstack.orchestration.v1 import software_deployment as _sd
from openstack.orchestration.v1 import stack as _stack
from openstack.orchestration.v1 import stack_environment as _stack_environment
from openstack.orchestration.v1 import stack_event as _stack_event
from openstack.orchestration.v1 import stack_files as _stack_files
from openstack.orchestration.v1 import stack_template as _stack_template
from openstack.orchestration.v1 import template as _template
from openstack import proxy
from openstack import resource
def read_env_and_templates(self, template_file=None, template_url=None, template_object=None, files=None, environment_files=None):
    """Read templates and environment content and prepares
        corresponding stack attributes

        :param string template_file: Path to the template.
        :param string template_url: URL of template.
        :param string template_object: URL to retrieve template object.
        :param dict files: dict of additional file content to include.
        :param environment_files: Paths to environment files to apply.

        :returns: Attributes dict to be set on the
            :class:`~openstack.orchestration.v1.stack.Stack`
        :rtype: dict
        """
    stack_attrs = dict()
    envfiles = dict()
    tpl_files = None
    if environment_files:
        envfiles, env = template_utils.process_multiple_environments_and_files(env_paths=environment_files)
        stack_attrs['environment'] = env
    if template_file or template_url or template_object:
        tpl_files, template = template_utils.get_template_contents(template_file=template_file, template_url=template_url, template_object=template_object, files=files)
        stack_attrs['template'] = template
        if tpl_files or envfiles:
            stack_attrs['files'] = dict(list(tpl_files.items()) + list(envfiles.items()))
    return stack_attrs