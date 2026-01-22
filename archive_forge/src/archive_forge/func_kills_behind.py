from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
def kills_behind(func):
    func.kills = 'behind'
    return func