import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_ppem_(self):
    location = self.cur_token_location_
    ppem_name = self.cur_token_
    value = self.expect_number_()
    setting = ast.SettingDefinition(ppem_name, value, location=location)
    return setting