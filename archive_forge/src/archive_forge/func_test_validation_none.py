from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validation_none(self):
    self.assertTrue(self.constraint.validate(None, self.ctx))