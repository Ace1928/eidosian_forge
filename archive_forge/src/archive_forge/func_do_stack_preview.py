import logging
import sys
from oslo_serialization import jsonutils
from oslo_utils import strutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import event_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_format
from heatclient.common import template_utils
from heatclient.common import utils
import heatclient.exc as exc
@utils.arg('-f', '--template-file', metavar='<FILE>', help=_('Path to the template.'))
@utils.arg('-e', '--environment-file', metavar='<FILE or URL>', help=_('Path to the environment, it can be specified multiple times.'), action='append')
@utils.arg('-u', '--template-url', metavar='<URL>', help=_('URL of template.'))
@utils.arg('-o', '--template-object', metavar='<URL>', help=_('URL to retrieve template object (e.g. from swift)'))
@utils.arg('-t', '--timeout', metavar='<TIMEOUT>', type=int, help=_('Stack creation timeout in minutes. This is only used during validation in preview.'))
@utils.arg('-r', '--enable-rollback', default=False, action='store_true', help=_('Enable rollback on failure. This option is not used during preview and exists only for symmetry with %(cmd)s.') % {'cmd': 'stack-create'})
@utils.arg('-P', '--parameters', metavar='<KEY1=VALUE1;KEY2=VALUE2...>', help=_('Parameter values used to preview the stack. This can be specified multiple times, or once with parameters separated by semicolon.'), action='append')
@utils.arg('-Pf', '--parameter-file', metavar='<KEY=FILE>', help=_('Parameter values from file used to create the stack. This can be specified multiple times. Parameter value would be the content of the file'), action='append')
@utils.arg('name', metavar='<STACK_NAME>', help=_('Name of the stack to preview.'))
@utils.arg('--tags', metavar='<TAG1,TAG2>', help=_('A list of tags to associate with the stack.'))
def do_stack_preview(hc, args):
    """Preview the stack."""
    show_deprecated('heat stack-preview', 'openstack stack create --dry-run')
    tpl_files, template = template_utils.get_template_contents(args.template_file, args.template_url, args.template_object, http.authenticated_fetcher(hc))
    env_files_list = []
    env_files, env = template_utils.process_multiple_environments_and_files(env_paths=args.environment_file, env_list_tracker=env_files_list)
    fields = {'stack_name': args.name, 'disable_rollback': not args.enable_rollback, 'timeout_mins': args.timeout, 'parameters': utils.format_all_parameters(args.parameters, args.parameter_file, args.template_file, args.template_url), 'template': template, 'files': dict(list(tpl_files.items()) + list(env_files.items())), 'environment': env}
    if env_files_list:
        fields['environment_files'] = env_files_list
    if args.tags:
        fields['tags'] = args.tags
    stack = hc.stacks.preview(**fields)
    formatters = {'description': utils.text_wrap_formatter, 'template_description': utils.text_wrap_formatter, 'stack_status_reason': utils.text_wrap_formatter, 'parameters': utils.json_formatter, 'outputs': utils.json_formatter, 'resources': utils.json_formatter, 'links': utils.link_formatter}
    utils.print_dict(stack.to_dict(), formatters=formatters)