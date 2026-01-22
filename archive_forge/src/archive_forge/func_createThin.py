from antlr4.Token import CommonToken
def createThin(self, type: int, text: str):
    t = CommonToken(type=type)
    t.text = text
    return t