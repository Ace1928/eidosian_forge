import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def do_preview_preproc(self):
    self.init_template_vars()
    template = self.clean_template(self.template)
    template = replace_tags(template, self.templatevars, self.templatevars)
    pp = TeXDimProc(template, self.options)
    usednodes = {}
    usededges = {}
    usedgraphs = {}
    counter = 0
    for node in self.main_graph.allnodes:
        name = node.name
        if node.attr.get('fixedsize', '') == 'true' or node.attr.get('style', '') in ['invis', 'invisible']:
            continue
        if node.attr.get('shape', '') == 'record':
            log.warning('Record nodes not supported in preprocessing mode: %s', name)
            continue
        texlbl = self.get_label(node)
        if texlbl:
            node.attr['texlbl'] = texlbl
            code = self.get_node_preproc_code(node)
            pp.add_snippet(name, code)
        usednodes[name] = node
    for edge in dotparsing.flatten(self.main_graph.alledges):
        if not edge.attr.get('label') and (not edge.attr.get('texlbl')) and (not edge.attr.get('headlabel')) and (not edge.attr.get('taillabel')):
            continue
        name = edge.src.name + edge.dst.name + str(counter)
        if is_multiline_label(edge):
            continue
        label = self.get_label(edge)
        headlabel = self.get_label(edge, 'headlabel', 'headtexlbl')
        taillabel = self.get_label(edge, 'taillabel', 'tailtexlbl')
        if label:
            name = edge.src.name + edge.dst.name + str(counter)
            edge.attr['texlbl'] = label
            code = self.get_edge_preproc_code(edge)
            pp.add_snippet(name, code)
        if headlabel:
            headlabel_name = name + 'headlabel'
            edge.attr['headtexlbl'] = headlabel
            code = self.get_edge_preproc_code(edge, 'headtexlbl')
            pp.add_snippet(headlabel_name, code)
        if taillabel:
            taillabel_name = name + 'taillabel'
            edge.attr['tailtexlbl'] = taillabel
            code = self.get_edge_preproc_code(edge, 'tailtexlbl')
            pp.add_snippet(taillabel_name, code)
        counter += 1
        usededges[name] = edge
    for graph in self.main_graph.allgraphs:
        if not graph.attr.get('label') and (not graph.attr.get('texlbl')):
            continue
        name = graph.name + str(counter)
        counter += 1
        label = self.get_label(graph)
        graph.attr['texlbl'] = label
        code = self.get_graph_preproc_code(graph)
        pp.add_snippet(name, code)
        usedgraphs[name] = graph
    ok = pp.process()
    if not ok:
        errormsg = 'Failed to preprocess the graph.\nIs the preview LaTeX package installed? ((Debian package preview-latex-style)\nTo see what happened, run dot2tex with the --debug option.\n'
        log.error(errormsg)
        sys.exit(1)
    for name, item in usednodes.items():
        if not item.attr.get('texlbl'):
            continue
        node = item
        hp, dp, wt = pp.texdims[name]
        if self.options.get('rawdim'):
            node.attr['width'] = wt
            node.attr['height'] = hp + dp
            node.attr['label'] = ' '
            node.attr['fixedsize'] = 'true'
            self.main_graph.allitems.append(node)
            continue
        xmargin, ymargin = self.get_margins(node)
        ht = hp + dp
        minwidth = float(item.attr.get('width') or DEFAULT_NODE_WIDTH)
        minheight = float(item.attr.get('height') or DEFAULT_NODE_HEIGHT)
        if self.options.get('nominsize'):
            width = wt + 2 * xmargin
            height = ht + 2 * ymargin
        else:
            if wt + 2 * xmargin < minwidth:
                width = minwidth
            else:
                width = wt + 2 * xmargin
            height = ht
            if hp + dp + 2 * ymargin < minheight:
                height = minheight
            else:
                height = ht + 2 * ymargin
        if item.attr.get('shape', '') in ['circle', 'Msquare', 'doublecircle', 'Mcircle']:
            if wt < height and width < height:
                width = height
            else:
                height = width
        node.attr['width'] = width
        node.attr['height'] = height
        node.attr['label'] = ' '
        node.attr['fixedsize'] = 'true'
        self.main_graph.allitems.append(node)
    for name, item in usededges.items():
        edge = item
        hp, dp, wt = pp.texdims[name]
        xmargin, ymargin = self.get_margins(edge)
        labelcode = '<<<table border="0" cellborder="0" cellpadding="0"><tr><td fixedsize="true" width="%s" height="%s">a</td></tr></table>>>'
        if 'texlbl' in edge.attr:
            edge.attr['label'] = labelcode % ((wt + 2 * xmargin) * 72, (hp + dp + 2 * ymargin) * 72)
        if 'tailtexlbl' in edge.attr:
            hp, dp, wt = pp.texdims[name + 'taillabel']
            edge.attr['taillabel'] = labelcode % ((wt + 2 * xmargin) * 72, (hp + dp + 2 * ymargin) * 72)
        if 'headtexlbl' in edge.attr:
            hp, dp, wt = pp.texdims[name + 'headlabel']
            edge.attr['headlabel'] = labelcode % ((wt + 2 * xmargin) * 72, (hp + dp + 2 * ymargin) * 72)
    for name, item in usedgraphs.items():
        graph = item
        hp, dp, wt = pp.texdims[name]
        xmargin, ymargin = self.get_margins(graph)
        labelcode = '<<<table border="0" cellborder="0" cellpadding="0"><tr><td fixedsize="true" width="%s" height="%s">a</td></tr></table>>>'
        graph.attr['label'] = labelcode % ((wt + 2 * xmargin) * 72, (hp + dp + 2 * ymargin) * 72)
    self.main_graph.attr['d2toutputformat'] = self.options.get('format', DEFAULT_OUTPUT_FORMAT)
    graphcode = str(self.main_graph)
    graphcode = graphcode.replace('<<<', '<<')
    graphcode = graphcode.replace('>>>', '>>')
    return graphcode