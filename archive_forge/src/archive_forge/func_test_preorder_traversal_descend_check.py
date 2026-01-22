import collections.abc
import pickle
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.block import block, block_dict
def test_preorder_traversal_descend_check(self):
    cdict, traversal = super(_TestActiveDictContainerBase, self).test_preorder_traversal_descend_check()
    cdict[1].deactivate()

    def descend(x):
        self.assertTrue(x._is_container)
        descend.seen.append(x)
        return not x._is_heterogeneous_container
    descend.seen = []
    order = list(pmo.preorder_traversal(cdict, active=True, descend=descend))
    self.assertEqual([None, '[0]', '[2]'], [c.name for c in order])
    self.assertEqual([id(cdict), id(cdict[0]), id(cdict[2])], [id(c) for c in order])
    if cdict.ctype._is_heterogeneous_container:
        self.assertEqual([None, '[0]', '[2]'], [c.name for c in descend.seen])
        self.assertEqual([id(cdict), id(cdict[0]), id(cdict[2])], [id(c) for c in descend.seen])
    else:
        self.assertEqual([None], [c.name for c in descend.seen])
        self.assertEqual([id(cdict)], [id(c) for c in descend.seen])

    def descend(x):
        self.assertTrue(x._is_container)
        descend.seen.append(x)
        return x.active and (not x._is_heterogeneous_container)
    descend.seen = []
    order = list(pmo.preorder_traversal(cdict, active=None, descend=descend))
    self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in order])
    self.assertEqual([id(cdict), id(cdict[0]), id(cdict[1]), id(cdict[2])], [id(c) for c in order])
    if cdict.ctype._is_heterogeneous_container:
        self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in descend.seen])
        self.assertEqual([id(cdict), id(cdict[0]), id(cdict[1]), id(cdict[2])], [id(c) for c in descend.seen])
    else:
        self.assertEqual([None, '[1]'], [c.name for c in descend.seen])
        self.assertEqual([id(cdict), id(cdict[1])], [id(c) for c in descend.seen])
    cdict[1].deactivate(shallow=False)

    def descend(x):
        descend.seen.append(x)
        return not x._is_heterogeneous_container
    descend.seen = []
    order = list(pmo.preorder_traversal(cdict, active=True, descend=descend))
    self.assertEqual([c.name for c in traversal if c.active], [c.name for c in order])
    self.assertEqual([id(c) for c in traversal if c.active], [id(c) for c in order])
    self.assertEqual([c.name for c in traversal if c.active and c._is_container], [c.name for c in descend.seen])
    self.assertEqual([id(c) for c in traversal if c.active and c._is_container], [id(c) for c in descend.seen])

    def descend(x):
        descend.seen.append(x)
        return x.active and (not x._is_heterogeneous_container)
    descend.seen = []
    order = list(pmo.preorder_traversal(cdict, active=None, descend=descend))
    self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in order])
    self.assertEqual([id(cdict), id(cdict[0]), id(cdict[1]), id(cdict[2])], [id(c) for c in order])
    if cdict.ctype._is_heterogeneous_container:
        self.assertEqual([None, '[0]', '[1]', '[2]'], [c.name for c in descend.seen])
        self.assertEqual([id(cdict), id(cdict[0]), id(cdict[1]), id(cdict[2])], [id(c) for c in descend.seen])
    else:
        self.assertEqual([None, '[1]'], [c.name for c in descend.seen])
        self.assertEqual([id(cdict), id(cdict[1])], [id(c) for c in descend.seen])
    cdict.deactivate()

    def descend(x):
        descend.seen.append(x)
        return True
    descend.seen = []
    order = list(pmo.preorder_traversal(cdict, active=True, descend=descend))
    self.assertEqual(len(descend.seen), 0)
    self.assertEqual(len(list(pmo.generate_names(cdict, active=True))), 0)

    def descend(x):
        descend.seen.append(x)
        return x.active
    descend.seen = []
    order = list(pmo.preorder_traversal(cdict, active=None, descend=descend))
    self.assertEqual(len(descend.seen), 1)
    self.assertIs(descend.seen[0], cdict)