import sys
from osc_lib.command import command
def print_software_deployment_output(data, name, out=sys.stdout, long=False):
    """Prints details of the software deployment for user consumption

    The format attempts to be valid yaml, but is primarily aimed at showing
    useful information to the user in a helpful layout.
    """
    if data is None:
        data = {}
    if name in ('deploy_stdout', 'deploy_stderr'):
        output = indent_and_truncate(data.get(name), spaces=4, truncate=not long, truncate_prefix='...', truncate_postfix='(truncated, view all with --long)')
        out.write('  %s: |\n%s\n' % (name, output))
    else:
        out.write('  %s: %s\n' % (name, data.get(name)))