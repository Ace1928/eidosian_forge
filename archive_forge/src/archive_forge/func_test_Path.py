import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
def test_Path(self):

    def norm(x):
        if cwd[1] == ':' and x[0] == '/':
            x = cwd[:2] + x
        return x.replace('/', os.path.sep)

    class ExamplePathLike:

        def __init__(self, path_str_or_bytes):
            self.path = path_str_or_bytes

        def __fspath__(self):
            return self.path

        def __str__(self):
            path_str = str(self.path)
            return f'{type(self).__name__}({path_str})'
    self.assertEqual(Path().domain_name(), 'Path')
    cwd = os.getcwd() + os.path.sep
    c = ConfigDict()
    c.declare('a', ConfigValue(None, Path()))
    self.assertEqual(c.a, None)
    c.a = '/a/b/c'
    self.assertTrue(os.path.sep in c.a)
    self.assertEqual(c.a, norm('/a/b/c'))
    c.a = b'/a/b/c'
    self.assertTrue(os.path.sep in c.a)
    self.assertEqual(c.a, norm('/a/b/c'))
    c.a = ExamplePathLike('/a/b/c')
    self.assertTrue(os.path.sep in c.a)
    self.assertEqual(c.a, norm('/a/b/c'))
    c.a = 'a/b/c'
    self.assertTrue(os.path.sep in c.a)
    self.assertEqual(c.a, norm(cwd + 'a/b/c'))
    c.a = b'a/b/c'
    self.assertTrue(os.path.sep in c.a)
    self.assertEqual(c.a, norm(cwd + 'a/b/c'))
    c.a = ExamplePathLike('a/b/c')
    self.assertTrue(os.path.sep in c.a)
    self.assertEqual(c.a, norm(cwd + 'a/b/c'))
    c.a = '${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.a)
    self.assertEqual(c.a, norm(cwd + 'a/b/c'))
    c.a = b'${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.a)
    self.assertEqual(c.a, norm(cwd + 'a/b/c'))
    c.a = ExamplePathLike('${CWD}/a/b/c')
    self.assertTrue(os.path.sep in c.a)
    self.assertEqual(c.a, norm(cwd + 'a/b/c'))
    c.a = None
    self.assertIs(c.a, None)
    c.declare('b', ConfigValue(None, Path('rel/path')))
    self.assertEqual(c.b, None)
    c.b = '/a/b/c'
    self.assertTrue(os.path.sep in c.b)
    self.assertEqual(c.b, norm('/a/b/c'))
    c.b = b'/a/b/c'
    self.assertTrue(os.path.sep in c.b)
    self.assertEqual(c.b, norm('/a/b/c'))
    c.b = ExamplePathLike('/a/b/c')
    self.assertTrue(os.path.sep in c.b)
    self.assertEqual(c.b, norm('/a/b/c'))
    c.b = 'a/b/c'
    self.assertTrue(os.path.sep in c.b)
    self.assertEqual(c.b, norm(cwd + 'rel/path/a/b/c'))
    c.b = b'a/b/c'
    self.assertTrue(os.path.sep in c.b)
    self.assertEqual(c.b, norm(cwd + 'rel/path/a/b/c'))
    c.b = ExamplePathLike('a/b/c')
    self.assertTrue(os.path.sep in c.b)
    self.assertEqual(c.b, norm(cwd + 'rel/path/a/b/c'))
    c.b = '${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.b)
    self.assertEqual(c.b, norm(cwd + 'a/b/c'))
    c.b = b'${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.b)
    self.assertEqual(c.b, norm(cwd + 'a/b/c'))
    c.b = ExamplePathLike('${CWD}/a/b/c')
    self.assertTrue(os.path.sep in c.b)
    self.assertEqual(c.b, norm(cwd + 'a/b/c'))
    c.b = None
    self.assertIs(c.b, None)
    c.declare('c', ConfigValue(None, Path('/my/dir')))
    self.assertEqual(c.c, None)
    c.c = '/a/b/c'
    self.assertTrue(os.path.sep in c.c)
    self.assertEqual(c.c, norm('/a/b/c'))
    c.c = b'/a/b/c'
    self.assertTrue(os.path.sep in c.c)
    self.assertEqual(c.c, norm('/a/b/c'))
    c.c = ExamplePathLike('/a/b/c')
    self.assertTrue(os.path.sep in c.c)
    self.assertEqual(c.c, norm('/a/b/c'))
    c.c = 'a/b/c'
    self.assertTrue(os.path.sep in c.c)
    self.assertEqual(c.c, norm('/my/dir/a/b/c'))
    c.c = b'a/b/c'
    self.assertTrue(os.path.sep in c.c)
    self.assertEqual(c.c, norm('/my/dir/a/b/c'))
    c.c = ExamplePathLike('a/b/c')
    self.assertTrue(os.path.sep in c.c)
    self.assertEqual(c.c, norm('/my/dir/a/b/c'))
    c.c = '${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.c)
    self.assertEqual(c.c, norm(cwd + 'a/b/c'))
    c.c = b'${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.c)
    self.assertEqual(c.c, norm(cwd + 'a/b/c'))
    c.c = ExamplePathLike('${CWD}/a/b/c')
    self.assertTrue(os.path.sep in c.c)
    self.assertEqual(c.c, norm(cwd + 'a/b/c'))
    c.c = None
    self.assertIs(c.c, None)
    c.declare('d_base', ConfigValue('${CWD}', str))
    c.declare('d', ConfigValue(None, Path(c.get('d_base'))))
    self.assertEqual(c.d, None)
    c.d = '/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm('/a/b/c'))
    c.d = b'/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm('/a/b/c'))
    c.d = ExamplePathLike('/a/b/c')
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm('/a/b/c'))
    c.d = 'a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    c.d = b'a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    c.d = ExamplePathLike('a/b/c')
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    c.d = '${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    c.d = b'${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    c.d = ExamplePathLike('${CWD}/a/b/c')
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    c.d_base = '/my/dir'
    c.d = '/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm('/a/b/c'))
    c.d = 'a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm('/my/dir/a/b/c'))
    c.d = '${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    c.d_base = 'rel/path'
    c.d = '/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm('/a/b/c'))
    c.d = b'/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm('/a/b/c'))
    c.d = ExamplePathLike('/a/b/c')
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm('/a/b/c'))
    c.d = 'a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'rel/path/a/b/c'))
    c.d = b'a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'rel/path/a/b/c'))
    c.d = ExamplePathLike('a/b/c')
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'rel/path/a/b/c'))
    c.d = '${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    c.d = b'${CWD}/a/b/c'
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    c.d = ExamplePathLike('${CWD}/a/b/c')
    self.assertTrue(os.path.sep in c.d)
    self.assertEqual(c.d, norm(cwd + 'a/b/c'))
    try:
        Path.SuppressPathExpansion = True
        c.d = '/a/b/c'
        self.assertTrue('/' in c.d)
        self.assertTrue('\\' not in c.d)
        self.assertEqual(c.d, '/a/b/c')
        c.d = b'/a/b/c'
        self.assertTrue('/' in c.d)
        self.assertTrue('\\' not in c.d)
        self.assertEqual(c.d, '/a/b/c')
        c.d = ExamplePathLike('/a/b/c')
        self.assertTrue('/' in c.d)
        self.assertTrue('\\' not in c.d)
        self.assertEqual(c.d, '/a/b/c')
        c.d = 'a/b/c'
        self.assertTrue('/' in c.d)
        self.assertTrue('\\' not in c.d)
        self.assertEqual(c.d, 'a/b/c')
        c.d = b'a/b/c'
        self.assertTrue('/' in c.d)
        self.assertTrue('\\' not in c.d)
        self.assertEqual(c.d, 'a/b/c')
        c.d = ExamplePathLike('a/b/c')
        self.assertTrue('/' in c.d)
        self.assertTrue('\\' not in c.d)
        self.assertEqual(c.d, 'a/b/c')
        c.d = '${CWD}/a/b/c'
        self.assertTrue('/' in c.d)
        self.assertTrue('\\' not in c.d)
        self.assertEqual(c.d, '${CWD}/a/b/c')
        c.d = b'${CWD}/a/b/c'
        self.assertTrue('/' in c.d)
        self.assertTrue('\\' not in c.d)
        self.assertEqual(c.d, '${CWD}/a/b/c')
        c.d = ExamplePathLike('${CWD}/a/b/c')
        self.assertTrue('/' in c.d)
        self.assertTrue('\\' not in c.d)
        self.assertEqual(c.d, '${CWD}/a/b/c')
    finally:
        Path.SuppressPathExpansion = False