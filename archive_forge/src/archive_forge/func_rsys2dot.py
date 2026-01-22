import os
import subprocess
import shutil
import tempfile
def rsys2dot(rsys, tex=False, rprefix='r', rref0=1, nodeparams='[label="{}",shape=diamond]', colors=('maroon', 'darkgreen'), penwidths=None, include_inactive=True):
    """
    Returns list of lines of DOT (graph description language)
    formatted graph.

    Parameters
    ==========
    rsys: ReactionSystem
    tex: bool (default False)
        If set True, output will be LaTeX formatted
    (Substance need to have latex_name attribute set)
    rprefix: string
        Reaction enumeration prefix, default: r
    rref0: integer
        Reaction enumeration initial counter value, default: 1
    nodeparams: string
        DOT formatted param list, default: [label={} shape=diamond]

    Returns
    =======
    list of lines of DOT representation of the graph representation.

    """
    lines = ['digraph "' + str(rsys.name) + '" {\n']
    ind = '  '
    if penwidths is None:
        penwidths = [1.0] * rsys.nr
    categories = rsys.categorize_substances(checks=())

    def add_substance(key):
        fc = 'black'
        if key in categories['depleted']:
            fc = colors[0]
        if key in categories['accumulated']:
            fc = colors[1]
        label = ('$%s$' if tex else '%s') % getattr(rsys.substances[key], 'latex_name' if tex else 'name')
        lines.append(ind + '"{key}" [fontcolor={fc} label="{lbl}"];\n'.format(key=key, fc=fc, lbl=label))
    for sk in rsys.substances:
        add_substance(sk)

    def add_vertex(key, num, reac, penwidth):
        snum = str(num) if num > 1 else ''
        fmt = ','.join(['label="{}"'.format(snum)] + (['penwidth={}'.format(penwidth)] if penwidth != 1 else []))
        lines.append(ind + '"{}" -> "{}" [color={},fontcolor={},{}];\n'.format(*((key, rid, colors[0], colors[0], fmt) if reac else (rid, key, colors[1], colors[1], fmt))))
    if include_inactive:
        reac_stoichs = rsys.all_reac_stoichs()
        prod_stoichs = rsys.all_prod_stoichs()
    else:
        reac_stoichs = rsys.active_reac_stoichs()
        prod_stoichs = rsys.active_prod_stoichs()
    for ri, rxn in enumerate(rsys.rxns):
        rid = rprefix + str(ri + rref0)
        lines.append(ind + '{')
        lines.append(ind * 2 + 'node ' + nodeparams.format(rxn.name or rid))
        lines.append(ind * 2 + rid)
        lines.append(ind + '}\n')
        for idx, key in enumerate(rsys.substances):
            num = reac_stoichs[ri, idx]
            if num == 0:
                continue
            add_vertex(key, num, True, penwidths[ri])
        for idx, key in enumerate(rsys.substances):
            num = prod_stoichs[ri, idx]
            if num == 0:
                continue
            add_vertex(key, num, False, penwidths[ri])
    lines.append('}\n')
    return lines