import configparser
from os.path import join, dirname
import textwrap
def get_cython_versions(setup_cfg=''):
    _cython_config = configparser.ConfigParser()
    if setup_cfg:
        _cython_config.read(setup_cfg)
    else:
        _cython_config.read(join(dirname(__file__), '..', '..', '..', 'setup.cfg'))
    cython_min = _cython_config['kivy']['cython_min']
    cython_max = _cython_config['kivy']['cython_max']
    cython_unsupported = _cython_config['kivy']['cython_exclude'].split(',')
    cython_requires = 'cython>={min_version},<={max_version},{exclusion}'.format(min_version=cython_min, max_version=cython_max, exclusion=','.join(('!=%s' % excl for excl in cython_unsupported)))
    return (cython_requires, cython_min, cython_max, cython_unsupported)