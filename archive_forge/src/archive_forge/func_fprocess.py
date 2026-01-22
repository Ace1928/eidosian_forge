import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def fprocess(infilep, outfilep):
    """
   Scans an input file for LA equations between double square brackets,
   e.g. [[ M3_mymatrix = M3_anothermatrix^-1 ]], and replaces the expression
   with a comment containing the equation followed by nested function calls
   that implement the equation as C code. A trailing semi-colon is appended.
   The equation within [[ ]] should NOT end with a semicolon as that will raise
   a ParseException. However, it is ok to have a semicolon after the right brackets.

   Other text in the file is unaltered.

   The arguments are file objects (NOT file names) opened for reading and
   writing, respectively.
   """
    pattern = '\\[\\[\\s*(.*?)\\s*\\]\\]'
    eqn = re.compile(pattern, re.DOTALL)
    s = infilep.read()

    def parser(mo):
        ccode = parse(mo.group(1))
        return '/* %s */\n%s;\nLAParserBufferReset();\n' % (mo.group(1), ccode)
    content = eqn.sub(parser, s)
    outfilep.write(content)