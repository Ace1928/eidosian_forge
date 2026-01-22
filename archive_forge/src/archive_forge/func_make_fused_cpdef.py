from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def make_fused_cpdef(self, orig_py_func, env, is_def):
    """
        This creates the function that is indexable from Python and does
        runtime dispatch based on the argument types. The function gets the
        arg tuple and kwargs dict (or None) and the defaults tuple
        as arguments from the Binding Fused Function's tp_call.
        """
    from . import TreeFragment, Code, UtilityCode
    fused_types = self._get_fused_base_types([arg.type for arg in self.node.args if arg.type.is_fused])
    context = {'memviewslice_cname': MemoryView.memviewslice_cname, 'func_args': self.node.args, 'n_fused': len(fused_types), 'min_positional_args': self.node.num_required_args - self.node.num_required_kw_args if is_def else sum((1 for arg in self.node.args if arg.default is None)), 'name': orig_py_func.entry.name}
    pyx_code = Code.PyxCodeWriter(context=context)
    decl_code = Code.PyxCodeWriter(context=context)
    decl_code.put_chunk(u'\n                cdef extern from *:\n                    void __pyx_PyErr_Clear "PyErr_Clear" ()\n                    type __Pyx_ImportNumPyArrayTypeIfAvailable()\n                    int __Pyx_Is_Little_Endian()\n            ')
    decl_code.indent()
    pyx_code.put_chunk(u'\n                def __pyx_fused_cpdef(signatures, args, kwargs, defaults, _fused_sigindex={}):\n                    # FIXME: use a typed signature - currently fails badly because\n                    #        default arguments inherit the types we specify here!\n\n                    cdef list search_list\n                    cdef dict sigindex_node\n\n                    dest_sig = [None] * {{n_fused}}\n\n                    if kwargs is not None and not kwargs:\n                        kwargs = None\n\n                    cdef Py_ssize_t i\n\n                    # instance check body\n            ')
    pyx_code.indent()
    pyx_code.named_insertion_point('imports')
    pyx_code.named_insertion_point('func_defs')
    pyx_code.named_insertion_point('local_variable_declarations')
    fused_index = 0
    default_idx = 0
    all_buffer_types = OrderedSet()
    seen_fused_types = set()
    for i, arg in enumerate(self.node.args):
        if arg.type.is_fused:
            arg_fused_types = arg.type.get_fused_types()
            if len(arg_fused_types) > 1:
                raise NotImplementedError('Determination of more than one fused base type per argument is not implemented.')
            fused_type = arg_fused_types[0]
        if arg.type.is_fused and fused_type not in seen_fused_types:
            seen_fused_types.add(fused_type)
            context.update(arg_tuple_idx=i, arg=arg, dest_sig_idx=fused_index, default_idx=default_idx)
            normal_types, buffer_types, pythran_types, has_object_fallback = self._split_fused_types(arg)
            self._unpack_argument(pyx_code)
            with pyx_code.indenter('while 1:'):
                if normal_types:
                    self._fused_instance_checks(normal_types, pyx_code, env)
                if buffer_types or pythran_types:
                    env.use_utility_code(Code.UtilityCode.load_cached('IsLittleEndian', 'ModuleSetupCode.c'))
                    self._buffer_checks(buffer_types, pythran_types, pyx_code, decl_code, arg.accept_none, env)
                if has_object_fallback:
                    pyx_code.context.update(specialized_type_name='object')
                    pyx_code.putln(self.match)
                else:
                    pyx_code.putln(self.no_match)
                pyx_code.putln('break')
            fused_index += 1
            all_buffer_types.update(buffer_types)
            all_buffer_types.update((ty.org_buffer for ty in pythran_types))
        if arg.default:
            default_idx += 1
    if all_buffer_types:
        self._buffer_declarations(pyx_code, decl_code, all_buffer_types, pythran_types)
        env.use_utility_code(Code.UtilityCode.load_cached('Import', 'ImportExport.c'))
        env.use_utility_code(Code.UtilityCode.load_cached('ImportNumPyArray', 'ImportExport.c'))
    self._fused_signature_index(pyx_code)
    pyx_code.put_chunk(u'\n                sigindex_matches = []\n                sigindex_candidates = [_fused_sigindex]\n\n                for dst_type in dest_sig:\n                    found_matches = []\n                    found_candidates = []\n                    # Make two separate lists: One for signature sub-trees\n                    #        with at least one definite match, and another for\n                    #        signature sub-trees with only ambiguous matches\n                    #        (where `dest_sig[i] is None`).\n                    if dst_type is None:\n                        for sn in sigindex_matches:\n                            found_matches.extend((<dict> sn).values())\n                        for sn in sigindex_candidates:\n                            found_candidates.extend((<dict> sn).values())\n                    else:\n                        for search_list in (sigindex_matches, sigindex_candidates):\n                            for sn in search_list:\n                                type_match = (<dict> sn).get(dst_type)\n                                if type_match is not None:\n                                    found_matches.append(type_match)\n                    sigindex_matches = found_matches\n                    sigindex_candidates = found_candidates\n                    if not (found_matches or found_candidates):\n                        break\n\n                candidates = sigindex_matches\n\n                if not candidates:\n                    raise TypeError("No matching signature found")\n                elif len(candidates) > 1:\n                    raise TypeError("Function call with ambiguous argument types")\n                else:\n                    return (<dict>signatures)[candidates[0]]\n            ')
    fragment_code = pyx_code.getvalue()
    from .Optimize import ConstantFolding
    fragment = TreeFragment.TreeFragment(fragment_code, level='module', pipeline=[ConstantFolding()])
    ast = TreeFragment.SetPosTransform(self.node.pos)(fragment.root)
    UtilityCode.declare_declarations_in_scope(decl_code.getvalue(), env.global_scope())
    ast.scope = env
    ast.analyse_declarations(env)
    py_func = ast.stats[-1]
    self.fragment_scope = ast.scope
    if isinstance(self.node, DefNode):
        py_func.specialized_cpdefs = self.nodes[:]
    else:
        py_func.specialized_cpdefs = [n.py_func for n in self.nodes]
    return py_func