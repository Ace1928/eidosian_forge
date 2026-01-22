from __future__ import annotations
import json
from os import makedirs
from os.path import exists, expanduser
from pymatgen.analysis.chemenv.utils.scripts_utils import strategies_class_lookup
from pymatgen.core import SETTINGS
@classmethod
def auto_load(cls, root_dir=None):
    """
        Autoload options.

        Args:
            root_dir:
        """
    if root_dir is None:
        home = expanduser('~')
        root_dir = f'{home}/.chemenv'
    config_file = f'{root_dir}/config.json'
    try:
        with open(config_file) as file:
            config_dict = json.load(file)
        return ChemEnvConfig(package_options=config_dict['package_options'])
    except OSError:
        print(f'Unable to load configuration from file {config_file!r} ...')
        print(' ... loading default configuration')
        return ChemEnvConfig()