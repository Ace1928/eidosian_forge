from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_loopControlInAsyncForElse(self):
    self.flakes('\n        async def read_data(db):\n            output = []\n            async for row in db.cursor():\n                output.append(row)\n            else:\n                continue\n            return output\n        ', m.ContinueOutsideLoop)
    self.flakes('\n        async def read_data(db):\n            output = []\n            async for row in db.cursor():\n                output.append(row)\n            else:\n                break\n            return output\n        ', m.BreakOutsideLoop)