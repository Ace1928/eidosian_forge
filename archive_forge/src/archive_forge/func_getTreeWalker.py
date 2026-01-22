from __future__ import absolute_import, division, unicode_literals
from .. import constants
from .._utils import default_etree
def getTreeWalker(treeType, implementation=None, **kwargs):
    """Get a TreeWalker class for various types of tree with built-in support

    :arg str treeType: the name of the tree type required (case-insensitive).
        Supported values are:

        * "dom": The xml.dom.minidom DOM implementation
        * "etree": A generic walker for tree implementations exposing an
          elementtree-like interface (known to work with ElementTree,
          cElementTree and lxml.etree).
        * "lxml": Optimized walker for lxml.etree
        * "genshi": a Genshi stream

    :arg implementation: A module implementing the tree type e.g.
        xml.etree.ElementTree or cElementTree (Currently applies to the "etree"
        tree type only).

    :arg kwargs: keyword arguments passed to the etree walker--for other
        walkers, this has no effect

    :returns: a TreeWalker class

    """
    treeType = treeType.lower()
    if treeType not in treeWalkerCache:
        if treeType == 'dom':
            from . import dom
            treeWalkerCache[treeType] = dom.TreeWalker
        elif treeType == 'genshi':
            from . import genshi
            treeWalkerCache[treeType] = genshi.TreeWalker
        elif treeType == 'lxml':
            from . import etree_lxml
            treeWalkerCache[treeType] = etree_lxml.TreeWalker
        elif treeType == 'etree':
            from . import etree
            if implementation is None:
                implementation = default_etree
            return etree.getETreeModule(implementation, **kwargs).TreeWalker
    return treeWalkerCache.get(treeType)