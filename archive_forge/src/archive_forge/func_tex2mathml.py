import docutils.utils.math.tex2unichar as tex2unichar
def tex2mathml(tex_math, inline=True):
    """Return string with MathML code corresponding to `tex_math`. 
    
    `inline`=True is for inline math and `inline`=False for displayed math.
    """
    mathml_tree = parse_latex_math(tex_math, inline=inline)
    return ''.join(mathml_tree.xml())