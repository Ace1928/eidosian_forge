import inspect
import functools
from enum import Enum
import torch.autograd
def hook_iterator(namespace):
    """
    Define a hook that is applied to all `__iter__` of metaclass `_DataPipeMeta`.

    This is done for the purpose of profiling and checking if an iterator is still valid.
    """

    def profiler_record_fn_context(datapipe):
        if not hasattr(datapipe, '_profile_name'):
            datapipe._profile_name = _generate_iterdatapipe_msg(datapipe, simplify_dp_name=True)
        return torch.autograd.profiler.record_function(datapipe._profile_name)

    class IteratorDecorator:
        """
        Wrap the iterator and modifying its `__next__` method.

        This decorator is applied to DataPipes of which `__iter__` method is NOT a generator function.
        Those `__iter__` method commonly returns `self` but not necessarily.
        """

        def __init__(self, iterator, datapipe, iterator_id, has_next_method):
            self.iterator = iterator
            self.datapipe = datapipe
            self.iterator_id = iterator_id
            self._profiler_enabled = torch.autograd._profiler_enabled()
            self.self_and_has_next_method = self.iterator is self.datapipe and has_next_method

        def __iter__(self):
            return self

        def _get_next(self):
            """Return next with logic related to iterator validity, profiler, and incrementation of samples yielded."""
            _check_iterator_valid(self.datapipe, self.iterator_id)
            result = next(self.iterator)
            if not self.self_and_has_next_method:
                self.datapipe._number_of_samples_yielded += 1
            return result

        def __next__(self):
            if self._profiler_enabled:
                with profiler_record_fn_context(self.datapipe):
                    return self._get_next()
            else:
                return self._get_next()

        def __getattr__(self, name):
            return getattr(self.iterator, name)
    func = namespace['__iter__']
    if inspect.isgeneratorfunction(func):

        @functools.wraps(func)
        def wrap_generator(*args, **kwargs):
            gen = func(*args, **kwargs)
            datapipe = args[0]
            if datapipe._fast_forward_iterator:
                it = datapipe._fast_forward_iterator
                datapipe._fast_forward_iterator = None
                datapipe._snapshot_state = _SnapshotState.Iterating
                while True:
                    try:
                        yield next(it)
                    except StopIteration:
                        return
            iterator_id = _set_datapipe_valid_iterator_id(datapipe)
            _profiler_enabled = torch.autograd._profiler_enabled()
            try:
                if _profiler_enabled:
                    with profiler_record_fn_context(datapipe):
                        response = gen.send(None)
                else:
                    response = gen.send(None)
                while True:
                    datapipe._number_of_samples_yielded += 1
                    request = (yield response)
                    if _profiler_enabled:
                        with profiler_record_fn_context(datapipe):
                            _check_iterator_valid(datapipe, iterator_id)
                            response = gen.send(request)
                    else:
                        _check_iterator_valid(datapipe, iterator_id)
                        response = gen.send(request)
            except StopIteration as e:
                return
            except Exception as e:
                datapipe = args[0]
                msg = 'thrown by __iter__ of'
                single_iterator_msg = 'single iterator per IterDataPipe constraint'
                if hasattr(e.args, '__len__'):
                    full_msg = f'{msg} {datapipe.__class__.__name__}({_generate_input_args_string(datapipe)})'
                    if len(e.args) == 0 or not isinstance(e.args[0], str):
                        e.args = (f'\nThis exception is {full_msg}',)
                    elif msg not in e.args[0] and single_iterator_msg not in e.args[0]:
                        e.args = (e.args[0] + f'\nThis exception is {full_msg}',) + e.args[1:]
                raise
        namespace['__iter__'] = wrap_generator
    else:
        if '__next__' in namespace:
            next_func = namespace['__next__']

            @functools.wraps(next_func)
            def wrap_next(*args, **kwargs):
                datapipe = args[0]
                if torch.autograd._profiler_enabled():
                    with profiler_record_fn_context(datapipe):
                        result = next_func(*args, **kwargs)
                else:
                    result = next_func(*args, **kwargs)
                datapipe._number_of_samples_yielded += 1
                return result
            namespace['__next__'] = wrap_next

        @functools.wraps(func)
        def wrap_iter(*args, **kwargs):
            iter_ret = func(*args, **kwargs)
            datapipe = args[0]
            datapipe._snapshot_state = _SnapshotState.Iterating
            if datapipe._fast_forward_iterator:
                iter_ret = datapipe._fast_forward_iterator
                datapipe._fast_forward_iterator = None
                return iter_ret
            iterator_id = _set_datapipe_valid_iterator_id(datapipe)
            return IteratorDecorator(iter_ret, datapipe, iterator_id, '__next__' in namespace)
        namespace['__iter__'] = wrap_iter