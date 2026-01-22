from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
def on(self, key=None, config=None):
    if not (key is None) ^ (config is None):
        raise ValueError('Must use exactly one of key, config')
    if key is not None:

        def add_to_keybinds(func):
            self.add(key, func)
            return func
        return add_to_keybinds
    else:

        def add_to_config(func):
            self.add_config_attr(config, func)
            return func
        return add_to_config