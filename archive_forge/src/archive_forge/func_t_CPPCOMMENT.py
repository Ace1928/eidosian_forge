def t_CPPCOMMENT(t):
    """//.*\\n"""
    t.lexer.lineno += 1
    return t