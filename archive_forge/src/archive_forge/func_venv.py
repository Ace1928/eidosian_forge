import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def venv(known_paths):
    global PREFIXES, ENABLE_USER_SITE
    env = os.environ
    if sys.platform == 'darwin' and '__PYVENV_LAUNCHER__' in env:
        executable = sys._base_executable = os.environ['__PYVENV_LAUNCHER__']
    else:
        executable = sys.executable
    exe_dir, _ = os.path.split(os.path.abspath(executable))
    site_prefix = os.path.dirname(exe_dir)
    sys._home = None
    conf_basename = 'pyvenv.cfg'
    candidate_confs = [conffile for conffile in (os.path.join(exe_dir, conf_basename), os.path.join(site_prefix, conf_basename)) if os.path.isfile(conffile)]
    if candidate_confs:
        virtual_conf = candidate_confs[0]
        system_site = 'true'
        with open(virtual_conf, encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip().lower()
                    value = value.strip()
                    if key == 'include-system-site-packages':
                        system_site = value.lower()
                    elif key == 'home':
                        sys._home = value
        sys.prefix = sys.exec_prefix = site_prefix
        addsitepackages(known_paths, [sys.prefix])
        if system_site == 'true':
            PREFIXES.insert(0, sys.prefix)
        else:
            PREFIXES = [sys.prefix]
            ENABLE_USER_SITE = False
    return known_paths