from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.error.ErrorListener import ProxyErrorListener, ConsoleErrorListener
def extractVersion(self, version):
    pos = version.find('.')
    major = version[0:pos]
    version = version[pos + 1:]
    pos = version.find('.')
    if pos == -1:
        pos = version.find('-')
    if pos == -1:
        pos = len(version)
    minor = version[0:pos]
    return (major, minor)