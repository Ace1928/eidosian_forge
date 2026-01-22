from pyomo.scripting.pyomo_parser import add_subparser, CustomHelpFormatter
from pyomo.common.deprecation import deprecated
@deprecated('Use of the pyomo install-extras is deprecated.The current recommended course of action is to manually install optional dependencies as needed.', version='5.7.1')
def install_extras(args=[], quiet=False):
    try:
        import pip
        pip_version = pip.__version__.split('.')
        for i, s in enumerate(pip_version):
            try:
                pip_version[i] = int(s)
            except:
                pass
        pip_version = tuple(pip_version)
    except ImportError:
        print("You must have 'pip' installed to run this script.")
        raise SystemExit
    cmd = ['--disable-pip-version-check', 'install', '--upgrade']
    if pip_version[0] >= 6:
        cmd.append('--no-cache-dir')
    else:
        cmd.append('--download-cache')
        cmd.append('')
    if not quiet:
        print(' ')
        print('-' * 60)
        print('Installation Output Logs')
        print('  (A summary will be printed below)')
        print('-' * 60)
        print(' ')
    results = {}
    for package in get_packages():
        if type(package) is tuple:
            package, pkg_import = package
        else:
            pkg_import = package
        try:
            pip.main(cmd + args + [package])
            __import__(pkg_import)
            results[package] = True
        except:
            results[package] = False
        try:
            pip.logger.consumers = []
        except AttributeError:
            pip.log.consumers = []
    if not quiet:
        print(' ')
        print(' ')
    print('-' * 60)
    print('Installation Summary')
    print('-' * 60)
    print(' ')
    for package, result in sorted(results.items()):
        if result:
            print('YES %s' % package)
        else:
            print('NO  %s' % package)