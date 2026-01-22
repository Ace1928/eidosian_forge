import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
Override this.  The renderers will pass the renderer,
        and the attribute name.  Algorithms can then backtrack up
        through all the stuff the renderer provides, including
        a correct stack of parent nodes.