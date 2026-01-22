import subprocess
def blahtexml(math_code, inline=True, reporter=None):
    """Convert LaTeX math code to MathML with blahtexml_

    .. _blahtexml: http://gva.noekeon.org/blahtexml/
    """
    options = ['--mathml', '--indented', '--spacing', 'moderate', '--mathml-encoding', 'raw', '--other-encoding', 'raw', '--doctype-xhtml+mathml', '--annotate-TeX']
    if inline:
        mathmode_arg = ''
    else:
        mathmode_arg = 'mode="display"'
        options.append('--displaymath')
    p = subprocess.Popen(['blahtexml'] + options, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    p.stdin.write(math_code.encode('utf8'))
    p.stdin.close()
    result = p.stdout.read().decode('utf8')
    err = p.stderr.read().decode('utf8')
    if result.find('<error>') >= 0:
        raise SyntaxError('\nMessage from external converter blahtexml:\n' + result[result.find('<message>') + 9:result.find('</message>')])
    if reporter and (err.find('**** Error') >= 0 or not result):
        reporter.error(err)
    start, end = (result.find('<markup>') + 9, result.find('</markup>'))
    result = '<math xmlns="http://www.w3.org/1998/Math/MathML"%s>\n%s</math>\n' % (mathmode_arg, result[start:end])
    return result