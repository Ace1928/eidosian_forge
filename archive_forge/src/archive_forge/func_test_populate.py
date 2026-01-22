import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def test_populate(self):
    pm = self.pm()
    pm.add_target_library_info('')
    pm.add_constant_merge_pass()
    pm.add_dead_arg_elimination_pass()
    pm.add_function_attrs_pass()
    pm.add_function_inlining_pass(225)
    pm.add_global_dce_pass()
    pm.add_global_optimizer_pass()
    pm.add_ipsccp_pass()
    pm.add_dead_code_elimination_pass()
    pm.add_cfg_simplification_pass()
    pm.add_gvn_pass()
    pm.add_instruction_combining_pass()
    pm.add_licm_pass()
    pm.add_sccp_pass()
    pm.add_sroa_pass()
    pm.add_type_based_alias_analysis_pass()
    pm.add_basic_alias_analysis_pass()
    pm.add_loop_rotate_pass()
    pm.add_region_info_pass()
    pm.add_scalar_evolution_aa_pass()
    pm.add_aggressive_dead_code_elimination_pass()
    pm.add_aa_eval_pass()
    pm.add_always_inliner_pass()
    pm.add_arg_promotion_pass(42)
    pm.add_break_critical_edges_pass()
    pm.add_dead_store_elimination_pass()
    pm.add_reverse_post_order_function_attrs_pass()
    pm.add_aggressive_instruction_combining_pass()
    pm.add_internalize_pass()
    pm.add_jump_threading_pass(7)
    pm.add_lcssa_pass()
    pm.add_loop_deletion_pass()
    pm.add_loop_extractor_pass()
    pm.add_single_loop_extractor_pass()
    pm.add_loop_strength_reduce_pass()
    pm.add_loop_simplification_pass()
    pm.add_loop_unroll_pass()
    pm.add_loop_unroll_and_jam_pass()
    pm.add_loop_unswitch_pass()
    pm.add_lower_atomic_pass()
    pm.add_lower_invoke_pass()
    pm.add_lower_switch_pass()
    pm.add_memcpy_optimization_pass()
    pm.add_merge_functions_pass()
    pm.add_merge_returns_pass()
    pm.add_partial_inlining_pass()
    pm.add_prune_exception_handling_pass()
    pm.add_reassociate_expressions_pass()
    pm.add_demote_register_to_memory_pass()
    pm.add_sink_pass()
    pm.add_strip_symbols_pass()
    pm.add_strip_dead_debug_info_pass()
    pm.add_strip_dead_prototypes_pass()
    pm.add_strip_debug_declare_pass()
    pm.add_strip_nondebug_symbols_pass()
    pm.add_tail_call_elimination_pass()
    pm.add_basic_aa_pass()
    pm.add_dependence_analysis_pass()
    pm.add_dot_call_graph_pass()
    pm.add_dot_cfg_printer_pass()
    pm.add_dot_dom_printer_pass()
    pm.add_dot_postdom_printer_pass()
    pm.add_globals_mod_ref_aa_pass()
    pm.add_iv_users_pass()
    pm.add_lazy_value_info_pass()
    pm.add_lint_pass()
    pm.add_module_debug_info_pass()
    pm.add_refprune_pass()
    pm.add_instruction_namer_pass()