from dill.source import getsource, getname, _wrap, likely_import
from dill.source import getimportable
from dill._dill import IS_PYPY
import sys
def test_importable():
    assert getimportable(add) == 'from %s import add\n' % __name__
    assert getimportable(squared) == 'from %s import squared\n' % __name__
    assert getimportable(Foo) == 'from %s import Foo\n' % __name__
    assert getimportable(Foo.bar) == 'from %s import bar\n' % __name__
    assert getimportable(_foo.bar) == 'from %s import bar\n' % __name__
    assert getimportable(None) == 'None\n'
    assert getimportable(100) == '100\n'
    assert getimportable(add, byname=False) == 'def add(x,y):\n  return x+y\n'
    assert getimportable(squared, byname=False) == 'squared = lambda x:x**2\n'
    assert getimportable(None, byname=False) == 'None\n'
    assert getimportable(Bar, byname=False) == 'class Bar:\n  pass\n'
    assert getimportable(Foo, byname=False) == 'class Foo(object):\n  def bar(self, x):\n    return x*x+x\n'
    assert getimportable(Foo.bar, byname=False) == 'def bar(self, x):\n  return x*x+x\n'
    assert getimportable(Foo.bar, byname=True) == 'from %s import bar\n' % __name__
    assert getimportable(Foo.bar, alias='memo', byname=True) == 'from %s import bar as memo\n' % __name__
    assert getimportable(Foo, alias='memo', byname=True) == 'from %s import Foo as memo\n' % __name__
    assert getimportable(squared, alias='memo', byname=True) == 'from %s import squared as memo\n' % __name__
    assert getimportable(squared, alias='memo', byname=False) == 'memo = squared = lambda x:x**2\n'
    assert getimportable(add, alias='memo', byname=False) == 'def add(x,y):\n  return x+y\n\nmemo = add\n'
    assert getimportable(None, alias='memo', byname=False) == 'memo = None\n'
    assert getimportable(100, alias='memo', byname=False) == 'memo = 100\n'
    assert getimportable(add, explicit=True) == 'from %s import add\n' % __name__
    assert getimportable(squared, explicit=True) == 'from %s import squared\n' % __name__
    assert getimportable(Foo, explicit=True) == 'from %s import Foo\n' % __name__
    assert getimportable(Foo.bar, explicit=True) == 'from %s import bar\n' % __name__
    assert getimportable(_foo.bar, explicit=True) == 'from %s import bar\n' % __name__
    assert getimportable(None, explicit=True) == 'None\n'
    assert getimportable(100, explicit=True) == '100\n'