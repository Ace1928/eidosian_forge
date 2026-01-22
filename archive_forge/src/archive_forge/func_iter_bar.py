from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def iter_bar(self, bar_prefix='', **kw):
    """Iterate through a list while updating a state bar.

        Examples
        --------
        >>> for username in logger.iter_bar(user=['tom', 'tim', 'lea']):
        >>>     # At every loop, logger.state['bars']['user'] is updated
        >>>     # to {index: i, total: 3, title:'user'}
        >>>     print (username)

        """
    if 'bar_message' in kw:
        bar_message = kw.pop('bar_message')
    else:
        bar_message = None
    bar, iterable = kw.popitem()
    if self.bar_is_ignored(bar) or self.iterable_is_too_short(iterable):
        return iterable
    bar = bar_prefix + bar
    if hasattr(iterable, '__len__'):
        self(**{bar + '__total': len(iterable)})

    def new_iterable():
        last_time = time.time()
        i = 0
        for i, it in enumerate(iterable):
            now_time = time.time()
            if i == 0 or now_time - last_time > self.min_time_interval:
                if bar_message is not None:
                    self(**{bar + '__message': bar_message(it)})
                self(**{bar + '__index': i})
                last_time = now_time
            yield it
        if self.bars[bar]['index'] != i:
            self(**{bar + '__index': i})
        self(**{bar + '__index': i + 1})
    return new_iterable()