import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_enclos_baseenv_set():
    env = rinterface.baseenv['new.env']()
    orig_enclosing_env = rinterface.baseenv.enclos
    enclosing_env = rinterface.baseenv['new.env']()
    env.enclos = enclosing_env
    assert isinstance(env.enclos, rinterface.SexpEnvironment)
    assert enclosing_env != env.enclos