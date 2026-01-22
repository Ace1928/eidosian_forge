import collections
import difflib
import io
import os
import tokenize
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.util import tf_inspect
def create_source_map(nodes, code, filepath):
    """Creates a source map between an annotated AST and the code it compiles to.

  Note: this function assumes nodes nodes, code and filepath correspond to the
  same code.

  Args:
    nodes: Iterable[ast.AST, ...], one or more AST modes.
    code: Text, the source code in which nodes are found.
    filepath: Text

  Returns:
    Dict[LineLocation, OriginInfo], mapping locations in code to locations
    indicated by origin annotations in node.
  """
    reparsed_nodes = parser.parse(code, preamble_len=0, single_node=False)
    for node in reparsed_nodes:
        resolve(node, code, filepath, node.lineno, node.col_offset)
    source_map = {}
    try:
        for before, after in ast_util.parallel_walk(nodes, reparsed_nodes):
            origin_info = anno.getanno(before, anno.Basic.ORIGIN, default=None)
            final_info = anno.getanno(after, anno.Basic.ORIGIN, default=None)
            if origin_info is None or final_info is None:
                continue
            line_loc = LineLocation(final_info.loc.filename, final_info.loc.lineno)
            existing_origin = source_map.get(line_loc)
            if existing_origin is not None:
                if existing_origin.loc.line_loc == origin_info.loc.line_loc:
                    if existing_origin.loc.lineno >= origin_info.loc.lineno:
                        continue
                if existing_origin.loc.col_offset <= origin_info.loc.col_offset:
                    continue
            source_map[line_loc] = origin_info
    except ValueError as err:
        new_msg = 'Inconsistent ASTs detected. This is a bug. Cause: \n'
        new_msg += str(err)
        new_msg += 'Diff:\n'
        for n, rn in zip(nodes, reparsed_nodes):
            nodes_str = pretty_printer.fmt(n, color=False, noanno=True)
            reparsed_nodes_str = pretty_printer.fmt(rn, color=False, noanno=True)
            diff = difflib.context_diff(nodes_str.split('\n'), reparsed_nodes_str.split('\n'), fromfile='Original nodes', tofile='Reparsed nodes', n=7)
            diff = '\n'.join(diff)
            new_msg += diff + '\n'
        raise ValueError(new_msg)
    return source_map