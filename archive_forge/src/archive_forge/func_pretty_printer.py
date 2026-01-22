import warnings
import functools
import locale
import weakref
import ctypes
import html
import textwrap
import llvmlite.binding as ll
import llvmlite.ir as llvmir
from abc import abstractmethod, ABCMeta
from numba.core import utils, config, cgutils
from numba.core.llvm_bindings import create_pass_manager_builder
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
from numba.core.runtime import rtsys
from numba.core.compiler_lock import require_global_compiler_lock
from numba.core.errors import NumbaInvalidConfigWarning
from numba.misc.inspection import disassemble_elf_to_cfg
from numba.misc.llvm_pass_timings import PassTimingsCollection
def pretty_printer(self, filename=None, view=None, render_format=None, highlight=True, interleave=False, strip_ir=False, show_key=True, fontsize=10):
    """
        "Pretty" prints the DOT graph of the CFG.
        For explanation of the parameters see the docstring for
        numba.core.dispatcher::inspect_cfg.
        """
    import graphviz as gv
    import re
    import json
    import inspect
    from llvmlite import binding as ll
    from numba.typed import List
    from types import SimpleNamespace
    from collections import defaultdict
    _default = False
    _highlight = SimpleNamespace(incref=_default, decref=_default, returns=_default, raises=_default, meminfo=_default, branches=_default, llvm_intrin_calls=_default, function_calls=_default)
    _interleave = SimpleNamespace(python=_default, lineinfo=_default)

    def parse_config(_config, kwarg):
        """ Parses the kwarg into a consistent format for use in configuring
            the Digraph rendering. _config is the configuration instance to
            update, kwarg is the kwarg on which to base the updates.
            """
        if isinstance(kwarg, bool):
            for attr in _config.__dict__:
                setattr(_config, attr, kwarg)
        elif isinstance(kwarg, dict):
            for k, v in kwarg.items():
                if k not in _config.__dict__:
                    raise ValueError('Unexpected key in kwarg: %s' % k)
                if isinstance(v, bool):
                    setattr(_config, k, v)
                else:
                    msg = 'Unexpected value for key: %s, got:%s'
                    raise ValueError(msg % (k, v))
        elif isinstance(kwarg, set):
            for item in kwarg:
                if item not in _config.__dict__:
                    raise ValueError('Unexpected key in kwarg: %s' % item)
                else:
                    setattr(_config, item, True)
        else:
            msg = 'Unhandled configuration type for kwarg %s'
            raise ValueError(msg % type(kwarg))
    parse_config(_highlight, highlight)
    parse_config(_interleave, interleave)
    cs = defaultdict(lambda: 'white')
    cs['marker'] = 'orange'
    cs['python'] = 'yellow'
    cs['truebr'] = 'green'
    cs['falsebr'] = 'red'
    cs['incref'] = 'cyan'
    cs['decref'] = 'turquoise'
    cs['raise'] = 'lightpink'
    cs['meminfo'] = 'lightseagreen'
    cs['return'] = 'purple'
    cs['llvm_intrin_calls'] = 'rosybrown'
    cs['function_calls'] = 'tomato'
    fn = self.cres.get_function(self.name)
    llvm_str = self.cres.get_llvm_str()

    def get_metadata(llvm_str):
        """ Gets the metadata entries from the LLVM IR, these look something
            like '!123 = INFORMATION'. Returns a map of metadata key to metadata
            value, i.e. from the example {'!123': INFORMATION}"""
        md = {}
        metadata_entry = re.compile('(^[!][0-9]+)(\\s+=\\s+.*)')
        for x in llvm_str.splitlines():
            match = metadata_entry.match(x)
            if match is not None:
                g = match.groups()
                if g is not None:
                    assert len(g) == 2
                    md[g[0]] = g[1]
        return md
    md = get_metadata(llvm_str)

    def init_digraph(name, fname, fontsize):
        cmax = 200
        if len(fname) > cmax:
            wstr = f'CFG output filename "{fname}" exceeds maximum supported length, it will be truncated.'
            warnings.warn(wstr, NumbaInvalidConfigWarning)
            fname = fname[:cmax]
        f = gv.Digraph(name, filename=fname)
        f.attr(rankdir='TB')
        f.attr('node', shape='none', fontsize='%s' % str(fontsize))
        return f
    f = init_digraph(self.name, self.name, fontsize)
    port_match = re.compile('.*{(.*)}.*')
    port_jmp_match = re.compile('.*<(.*)>(.*)')
    metadata_marker = re.compile('.*!dbg\\s+(![0-9]+).*')
    location_expr = '.*!DILocation\\(line:\\s+([0-9]+),\\s+column:\\s+([0-9]),.*'
    location_entry = re.compile(location_expr)
    dbg_value = re.compile('.*call void @llvm.dbg.value.*')
    nrt_incref = re.compile('@NRT_incref\\b')
    nrt_decref = re.compile('@NRT_decref\\b')
    nrt_meminfo = re.compile('@NRT_MemInfo')
    ll_intrin_calls = re.compile('.*call.*@llvm\\..*')
    ll_function_call = re.compile('.*call.*@.*')
    ll_raise = re.compile('store .*\\!numba_exception_output.*')
    ll_return = re.compile('ret i32 [^1],?.*')

    def wrap(s):
        return textwrap.wrap(s, width=120, subsequent_indent='... ')

    def clean(s):
        n = 300
        if len(s) > n:
            hs = str(hash(s))
            s = '{}...<hash={}>'.format(s[:n], hs)
        s = html.escape(s)
        s = s.replace('\\{', '&#123;')
        s = s.replace('\\}', '&#125;')
        s = s.replace('\\', '&#92;')
        s = s.replace('%', '&#37;')
        s = s.replace('!', '&#33;')
        return s
    node_ids = {}
    edge_ids = {}
    if _interleave.python:
        src_code, firstlineno = inspect.getsourcelines(self.py_func)
    raw_dot = ll.get_function_cfg(fn).replace('\\l...', '')
    json_bytes = gv.Source(raw_dot).pipe(format='dot_json')
    jzon = json.loads(json_bytes.decode('utf-8'))
    idc = 0
    for obj in jzon['objects']:
        cur_line, cur_col = (-1, -1)
        label = obj['label']
        name = obj['name']
        gvid = obj['_gvid']
        node_ids[gvid] = name
        label = label[1:-1]
        lines = label.split('\\l')
        new_lines = []
        col_span = 1
        port_line = ''
        matched = port_match.match(lines[-1])
        sliced_lines = lines
        if matched is not None:
            ports = matched.groups()[0]
            ports_tokens = ports.split('|')
            col_span = len(ports_tokens)
            tdfmt = '<td BGCOLOR="{}" BORDER="1" ALIGN="center" PORT="{}">{}</td>'
            tbl_data = []
            if _highlight.branches:
                colors = {'T': cs['truebr'], 'F': cs['falsebr']}
            else:
                colors = {}
            for tok in ports_tokens:
                target, value = port_jmp_match.match(tok).groups()
                color = colors.get(value, 'white')
                tbl_data.append(tdfmt.format(color, target, value))
            port_line = ''.join(tbl_data)
            sliced_lines = lines[:-1]
        fmtheader = '<tr><td BGCOLOR="{}" BORDER="1" ALIGN="left" COLSPAN="{}">{}</td></tr>'
        new_lines.append(fmtheader.format(cs['default'], col_span, clean(sliced_lines[0].strip())))
        fmt = '<tr><td BGCOLOR="{}" BORDER="0" ALIGN="left" COLSPAN="{}">{}</td></tr>'

        def metadata_interleave(l, new_lines):
            """
                Search line `l` for metadata associated with python or line info
                and inject it into `new_lines` if requested.
                """
            matched = metadata_marker.match(l)
            if matched is not None:
                g = matched.groups()
                if g is not None:
                    assert len(g) == 1, g
                    marker = g[0]
                    debug_data = md.get(marker, None)
                    if debug_data is not None:
                        ld = location_entry.match(debug_data)
                        if ld is not None:
                            assert len(ld.groups()) == 2, ld
                            line, col = ld.groups()
                            if line != cur_line or col != cur_col:
                                if _interleave.lineinfo:
                                    mfmt = 'Marker %s, Line %s, column %s'
                                    mark_line = mfmt % (marker, line, col)
                                    ln = fmt.format(cs['marker'], col_span, clean(mark_line))
                                    new_lines.append(ln)
                                if _interleave.python:
                                    lidx = int(line) - (firstlineno + 1)
                                    source_line = src_code[lidx + 1]
                                    ln = fmt.format(cs['python'], col_span, clean(source_line))
                                    new_lines.append(ln)
                                return (line, col)
        for l in sliced_lines[1:]:
            if dbg_value.match(l):
                continue
            if _interleave.lineinfo or _interleave.python:
                updated_lineinfo = metadata_interleave(l, new_lines)
                if updated_lineinfo is not None:
                    cur_line, cur_col = updated_lineinfo
            if _highlight.incref and nrt_incref.search(l):
                colour = cs['incref']
            elif _highlight.decref and nrt_decref.search(l):
                colour = cs['decref']
            elif _highlight.meminfo and nrt_meminfo.search(l):
                colour = cs['meminfo']
            elif _highlight.raises and ll_raise.search(l):
                colour = cs['raise']
            elif _highlight.returns and ll_return.search(l):
                colour = cs['return']
            elif _highlight.llvm_intrin_calls and ll_intrin_calls.search(l):
                colour = cs['llvm_intrin_calls']
            elif _highlight.function_calls and ll_function_call.search(l):
                colour = cs['function_calls']
            else:
                colour = cs['default']
            if colour is not cs['default'] or not strip_ir:
                for x in wrap(clean(l)):
                    new_lines.append(fmt.format(colour, col_span, x))
        if port_line:
            new_lines.append('<tr>{}</tr>'.format(port_line))
        dat = ''.join(new_lines)
        if dat:
            tab = '<table id="%s" BORDER="1" CELLBORDER="0" CELLPADDING="0" CELLSPACING="0">%s</table>' % (idc, dat)
            label = '<{}>'.format(tab)
        else:
            label = ''
        f.node(name, label=label)
    if 'edges' in jzon:
        for edge in jzon['edges']:
            gvid = edge['_gvid']
            tp = edge.get('tailport', None)
            edge_ids[gvid] = (edge['head'], edge['tail'], tp)
    for gvid, edge in edge_ids.items():
        tail = node_ids[edge[1]]
        head = node_ids[edge[0]]
        port = edge[2]
        if port is not None:
            tail += ':%s' % port
        f.edge(tail, head)
    if show_key:
        key_tab = []
        for k, v in cs.items():
            key_tab.append('<tr><td BGCOLOR="{}" BORDER="0" ALIGN="center">{}</td></tr>'.format(v, k))
        f.node('Key', label='<<table BORDER="1" CELLBORDER="1" CELLPADDING="2" CELLSPACING="1"><tr><td BORDER="0">Key:</td></tr>{}</table>>'.format(''.join(key_tab)))
    if filename is not None or view is not None:
        f.render(filename=filename, view=view, format=render_format)
    return f.pipe(format='svg')