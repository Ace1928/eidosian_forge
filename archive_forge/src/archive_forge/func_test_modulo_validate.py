from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_modulo_validate(self):
    r = constraints.Modulo(step=2, offset=1, description='a modulo')
    r.validate(1)
    r.validate(3)
    r.validate(5)
    r.validate(777777)
    r = constraints.Modulo(step=111, offset=0, description='a modulo')
    r.validate(111)
    r.validate(222)
    r.validate(444)
    r.validate(1110)
    r = constraints.Modulo(step=111, offset=11, description='a modulo')
    r.validate(122)
    r.validate(233)
    r.validate(1121)
    r = constraints.Modulo(step=-2, offset=-1, description='a modulo')
    r.validate(-1)
    r.validate(-3)
    r.validate(-5)
    r.validate(-777777)
    r = constraints.Modulo(step=-2, offset=0, description='a modulo')
    r.validate(-2)
    r.validate(-4)
    r.validate(-8888888)