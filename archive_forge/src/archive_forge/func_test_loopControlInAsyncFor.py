from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_loopControlInAsyncFor(self):
    self.flakes("\n        async def read_data(db):\n            output = []\n            async for row in db.cursor():\n                if row[0] == 'skip':\n                    continue\n                output.append(row)\n            return output\n        ")
    self.flakes("\n        async def read_data(db):\n            output = []\n            async for row in db.cursor():\n                if row[0] == 'stop':\n                    break\n                output.append(row)\n            return output\n        ")