from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def puppet_runner(module):

    def _prepare_base_cmd():
        _tout_cmd = module.get_bin_path('timeout', False)
        if _tout_cmd:
            cmd = ['timeout', '-s', '9', module.params['timeout'], _puppet_cmd(module)]
        else:
            cmd = ['puppet']
        return cmd

    def noop_func(v):
        return ['--noop'] if module.check_mode or v else ['--no-noop']
    _logdest_map = {'syslog': ['--logdest', 'syslog'], 'all': ['--logdest', 'syslog', '--logdest', 'console']}

    @cmd_runner_fmt.unpack_args
    def execute_func(execute, manifest):
        if execute:
            return ['--execute', execute]
        else:
            return [manifest]
    runner = CmdRunner(module, command=_prepare_base_cmd(), path_prefix=_PUPPET_PATH_PREFIX, arg_formats=dict(_agent_fixed=cmd_runner_fmt.as_fixed(['agent', '--onetime', '--no-daemonize', '--no-usecacheonfailure', '--no-splay', '--detailed-exitcodes', '--verbose', '--color', '0']), _apply_fixed=cmd_runner_fmt.as_fixed(['apply', '--detailed-exitcodes']), puppetmaster=cmd_runner_fmt.as_opt_val('--server'), show_diff=cmd_runner_fmt.as_bool('--show-diff'), confdir=cmd_runner_fmt.as_opt_val('--confdir'), environment=cmd_runner_fmt.as_opt_val('--environment'), tags=cmd_runner_fmt.as_func(lambda v: ['--tags', ','.join(v)]), skip_tags=cmd_runner_fmt.as_func(lambda v: ['--skip_tags', ','.join(v)]), certname=cmd_runner_fmt.as_opt_eq_val('--certname'), noop=cmd_runner_fmt.as_func(noop_func), use_srv_records=cmd_runner_fmt.as_map({True: '--usr_srv_records', False: '--no-usr_srv_records'}), logdest=cmd_runner_fmt.as_map(_logdest_map, default=[]), modulepath=cmd_runner_fmt.as_opt_eq_val('--modulepath'), _execute=cmd_runner_fmt.as_func(execute_func), summarize=cmd_runner_fmt.as_bool('--summarize'), debug=cmd_runner_fmt.as_bool('--debug'), verbose=cmd_runner_fmt.as_bool('--verbose')), check_rc=False)
    return runner