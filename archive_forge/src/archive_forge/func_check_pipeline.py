from __future__ import print_function
from functools import wraps
import warnings
from .export_utils import expr_to_tree, generate_pipeline_code
from deap import creator
from stopit import threading_timeoutable, TimeoutException
@wraps(func)
def check_pipeline(self, *args, **kwargs):
    bad_pipeline = True
    num_test = 0
    while bad_pipeline and num_test < NUM_TESTS:
        args = [self._toolbox.clone(arg) if isinstance(arg, creator.Individual) else arg for arg in args]
        try:
            if func.__name__ == '_generate':
                expr = []
            else:
                expr = tuple(args)
            pass_gen = False
            num_test_expr = 0
            while not pass_gen and num_test_expr < int(NUM_TESTS / 2):
                try:
                    expr = func(self, *args, **kwargs)
                    pass_gen = True
                except:
                    num_test_expr += 1
                    pass
            expr_tuple = expr if isinstance(expr, tuple) else (expr,)
            if func.__name__ == '_mate_operator':
                expr_tuple = expr_tuple[0:2]
            for expr_test in expr_tuple:
                pipeline_code = generate_pipeline_code(expr_to_tree(expr_test, self._pset), self.operators)
                sklearn_pipeline = eval(pipeline_code, self.operators_context)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    time_limited_call(sklearn_pipeline.fit, self.pretest_X, self.pretest_y, timeout=MAX_EVAL_SECS)
                bad_pipeline = False
        except BaseException as e:
            message = '_pre_test decorator: {fname}: num_test={n} {e}.'.format(n=num_test, fname=func.__name__, e=e)
            self._update_pbar(pbar_num=0, pbar_msg=message)
        finally:
            num_test += 1
    return expr