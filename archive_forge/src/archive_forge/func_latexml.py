import subprocess
def latexml(math_code, reporter=None):
    """Convert LaTeX math code to MathML with LaTeXML_

    .. _LaTeXML: http://dlmf.nist.gov/LaTeXML/
    """
    p = subprocess.Popen(['latexml', '-', '--inputencoding=utf8'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    p.stdin.write((document_template % math_code).encode('utf8'))
    p.stdin.close()
    latexml_code = p.stdout.read()
    latexml_err = p.stderr.read().decode('utf8')
    if reporter and (latexml_err.find('Error') >= 0 or not latexml_code):
        reporter.error(latexml_err)
    post_p = subprocess.Popen(['latexmlpost', '-', '--nonumbersections', '--format=xhtml', '--'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    post_p.stdin.write(latexml_code)
    post_p.stdin.close()
    result = post_p.stdout.read().decode('utf8')
    post_p_err = post_p.stderr.read().decode('utf8')
    if reporter and (post_p_err.find('Error') >= 0 or not result):
        reporter.error(post_p_err)
    start, end = (result.find('<math'), result.find('</math>') + 7)
    result = result[start:end]
    if 'class="ltx_ERROR' in result:
        raise SyntaxError(result)
    return result