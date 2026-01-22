from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow.compat.v2 as tf
from absl import flags
from absl.testing import absltest
from keras.src.testing_infra import keras_doctest_lib
import doctest  # noqa: E402
def load_tests(unused_loader, tests, unused_ignore):
    """Loads all the tests in the docstrings and runs them."""
    tf_modules = find_modules()
    if FLAGS.module:
        tf_modules = filter_on_submodules(tf_modules, FLAGS.module)
    if FLAGS.list:
        print('**************************************************')
        for mod in tf_modules:
            print(mod.__name__)
        print('**************************************************')
        return tests
    if FLAGS.file:
        tf_modules = get_module_and_inject_docstring(FLAGS.file)
    for module in tf_modules:
        testcase = TfTestCase()
        tests.addTests(doctest.DocTestSuite(module, test_finder=doctest.DocTestFinder(exclude_empty=False), extraglobs={'tf': tf, 'np': np, 'os': os}, setUp=testcase.set_up, tearDown=testcase.tear_down, checker=keras_doctest_lib.KerasDoctestOutputChecker(), optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL | doctest.DONT_ACCEPT_BLANKLINE))
    return tests