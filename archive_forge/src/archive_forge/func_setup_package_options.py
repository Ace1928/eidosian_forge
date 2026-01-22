from __future__ import annotations
import json
from os import makedirs
from os.path import exists, expanduser
from pymatgen.analysis.chemenv.utils.scripts_utils import strategies_class_lookup
from pymatgen.core import SETTINGS
def setup_package_options(self):
    """Setup the package options."""
    self.package_options = self.DEFAULT_PACKAGE_OPTIONS
    print('Choose between the following strategies : ')
    strategies = list(strategies_class_lookup)
    for idx, strategy in enumerate(strategies, start=1):
        print(f' <{idx}> : {strategy}')
    test = input(' ... ')
    self.package_options['default_strategy'] = {'strategy': strategies[int(test) - 1], 'strategy_options': {}}
    strategy_class = strategies_class_lookup[strategies[int(test) - 1]]
    if len(strategy_class.STRATEGY_OPTIONS) > 0:
        for option, option_dict in strategy_class.STRATEGY_OPTIONS.items():
            while True:
                print(f'  => Enter value for option {option!r} (<ENTER> for default = {option_dict['default']})\n')
                print('     Valid options are :\n')
                print(f'       {option_dict['type'].allowed_values}')
                test = input('     Your choice : ')
                if test == '':
                    self.package_options['default_strategy']['strategy_options'][option] = option_dict['type'](strategy_class.STRATEGY_OPTIONS[option]['default'])
                    break
                try:
                    self.package_options['default_strategy']['strategy_options'][option] = option_dict['type'](test)
                    break
                except ValueError:
                    print(f'Wrong input for option={option!r}')