import os
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
import pyomo.repn.plugins.ampl.ampl_ as ampl_
import pyomo.repn.plugins.nl_writer as nl_writer
def test_obj_con_cache(self):
    if self._nl_version != 'nl_v1':
        self.skipTest(f'test not applicable to writer {self._nl_version}')
    model = ConcreteModel()
    model.x = Var()
    model.c = Constraint(expr=model.x ** 2 >= 1)
    model.obj = Objective(expr=model.x ** 2)
    with TempfileManager.new_context() as TMP:
        nl_file = TMP.create_tempfile(suffix='.nl')
        model.write(nl_file, format=self._nl_version)
        self.assertFalse(hasattr(model, '_repn'))
        with open(nl_file) as FILE:
            nl_ref = FILE.read()
        nl_file = TMP.create_tempfile(suffix='.nl')
        model._gen_obj_repn = True
        model.write(nl_file, format=self._nl_version)
        self.assertEqual(len(model._repn), 1)
        self.assertIn(model.obj, model._repn)
        obj_repn = model._repn[model.obj]
        with open(nl_file) as FILE:
            nl_test = FILE.read()
        self.assertEqual(nl_ref, nl_test)
        nl_file = TMP.create_tempfile(suffix='.nl')
        del model._repn
        model._gen_obj_repn = None
        model._gen_con_repn = True
        model.write(nl_file, format=self._nl_version)
        self.assertEqual(len(model._repn), 1)
        self.assertIn(model.c, model._repn)
        c_repn = model._repn[model.c]
        with open(nl_file) as FILE:
            nl_test = FILE.read()
        self.assertEqual(nl_ref, nl_test)
        nl_file = TMP.create_tempfile(suffix='.nl')
        del model._repn
        model._gen_obj_repn = True
        model._gen_con_repn = True
        model.write(nl_file, format=self._nl_version)
        self.assertEqual(len(model._repn), 2)
        self.assertIn(model.obj, model._repn)
        self.assertIn(model.c, model._repn)
        obj_repn = model._repn[model.obj]
        c_repn = model._repn[model.c]
        with open(nl_file) as FILE:
            nl_test = FILE.read()
        self.assertEqual(nl_ref, nl_test)
        nl_file = TMP.create_tempfile(suffix='.nl')
        model._gen_obj_repn = None
        model._gen_con_repn = None
        model.write(nl_file, format=self._nl_version)
        self.assertEqual(len(model._repn), 2)
        self.assertIn(model.obj, model._repn)
        self.assertIn(model.c, model._repn)
        self.assertIs(obj_repn, model._repn[model.obj])
        self.assertIs(c_repn, model._repn[model.c])
        with open(nl_file) as FILE:
            nl_test = FILE.read()
        self.assertEqual(nl_ref, nl_test)
        nl_file = TMP.create_tempfile(suffix='.nl')
        model._gen_obj_repn = True
        model._gen_con_repn = True
        model.write(nl_file, format=self._nl_version)
        self.assertEqual(len(model._repn), 2)
        self.assertIn(model.obj, model._repn)
        self.assertIn(model.c, model._repn)
        self.assertIsNot(obj_repn, model._repn[model.obj])
        self.assertIsNot(c_repn, model._repn[model.c])
        obj_repn = model._repn[model.obj]
        c_repn = model._repn[model.c]
        with open(nl_file) as FILE:
            nl_test = FILE.read()
        self.assertEqual(nl_ref, nl_test)
        nl_file = TMP.create_tempfile(suffix='.nl')
        model._gen_obj_repn = False
        model._gen_con_repn = False
        try:

            def dont_call_gsr(*args, **kwargs):
                self.fail('generate_standard_repn should not be called')
            ampl_.generate_standard_repn = dont_call_gsr
            model.write(nl_file, format=self._nl_version)
        finally:
            ampl_.generate_standard_repn = gsr
        self.assertEqual(len(model._repn), 2)
        self.assertIn(model.obj, model._repn)
        self.assertIn(model.c, model._repn)
        self.assertIs(obj_repn, model._repn[model.obj])
        self.assertIs(c_repn, model._repn[model.c])
        with open(nl_file) as FILE:
            nl_test = FILE.read()
        self.assertEqual(nl_ref, nl_test)
        model._repn[model.c] = c_repn = gsr(model.c.body, quadratic=True)
        model._repn[model.obj] = obj_repn = gsr(model.obj.expr, quadratic=True)
        nl_file = TMP.create_tempfile(suffix='.nl')
        try:

            def dont_call_gsr(*args, **kwargs):
                self.fail('generate_standard_repn should not be called')
            ampl_.generate_standard_repn = dont_call_gsr
            model.write(nl_file, format=self._nl_version)
        finally:
            ampl_.generate_standard_repn = gsr
        self.assertEqual(len(model._repn), 2)
        self.assertIn(model.obj, model._repn)
        self.assertIn(model.c, model._repn)
        self.assertIs(obj_repn, model._repn[model.obj])
        self.assertIs(c_repn, model._repn[model.c])
        with open(nl_file) as FILE:
            nl_test = FILE.read()
        self.assertEqual(nl_ref, nl_test)