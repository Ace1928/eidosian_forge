import inspect
import os
import threading
import time
import warnings
from modin.config import Engine, ProgressBar
def call_progress_bar(result_parts, line_no):
    """
    Attach a progress bar to given `result_parts`.

    The progress bar is expected to be shown in a Jupyter Notebook cell.

    Parameters
    ----------
    result_parts : list of list of object refs (futures)
        Objects which are being computed for which progress is requested.
    line_no : int
        Line number in the call stack which we're displaying progress for.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            from tqdm.autonotebook import tqdm as tqdm_notebook
        except ImportError:
            raise ImportError('Please pip install tqdm to use the progress bar')
        from IPython import get_ipython
    try:
        cell_no = get_ipython().execution_count
    except AttributeError:
        return
    pbar_id = f'{cell_no}-{line_no}'
    futures = [block for row in result_parts for partition in row for block in partition.list_of_blocks]
    bar_format = '{l_bar}{bar}{r_bar}' if 'DEBUG_PROGRESS_BAR' in os.environ and os.environ['DEBUG_PROGRESS_BAR'] == 'True' else '{desc}: {percentage:3.0f}%{bar} Elapsed time: {elapsed}, estimated remaining time: {remaining}'
    bar_lock.acquire()
    if pbar_id in progress_bars:
        if hasattr(progress_bars[pbar_id], 'container'):
            if hasattr(progress_bars[pbar_id].container.children[0], 'max'):
                index = 0
            else:
                index = 1
            progress_bars[pbar_id].container.children[index].max = progress_bars[pbar_id].container.children[index].max + len(futures)
        progress_bars[pbar_id].total = progress_bars[pbar_id].total + len(futures)
        progress_bars[pbar_id].refresh()
    else:
        progress_bars[pbar_id] = tqdm_notebook(total=len(futures), desc='Estimated completion of line ' + str(line_no), bar_format=bar_format)
    bar_lock.release()
    threading.Thread(target=_show_time_updates, args=(progress_bars[pbar_id],)).start()
    modin_engine = Engine.get()
    engine_wrapper = None
    if modin_engine == 'Ray':
        from modin.core.execution.ray.common.engine_wrapper import RayWrapper
        engine_wrapper = RayWrapper
    elif modin_engine == 'Unidist':
        from modin.core.execution.unidist.common.engine_wrapper import UnidistWrapper
        engine_wrapper = UnidistWrapper
    else:
        raise NotImplementedError(f'ProgressBar feature is not supported for {modin_engine} engine.')
    for i in range(1, len(futures) + 1):
        engine_wrapper.wait(futures, num_returns=i)
        progress_bars[pbar_id].update(1)
        progress_bars[pbar_id].refresh()
    if progress_bars[pbar_id].n == progress_bars[pbar_id].total:
        progress_bars[pbar_id].close()