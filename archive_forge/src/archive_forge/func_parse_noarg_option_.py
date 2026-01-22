import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_noarg_option_(self):
    location = self.cur_token_location_
    name = self.cur_token_
    value = True
    setting = ast.SettingDefinition(name, value, location=location)
    return setting