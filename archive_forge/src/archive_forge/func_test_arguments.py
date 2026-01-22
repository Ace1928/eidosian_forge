from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_arguments(self):
    """
        L{task.react} passes the elements of the list it is passed as
        positional arguments to the function it is passed.
        """
    args = []

    async def main(reactor, x, y, z):
        args.extend((x, y, z))
        return await defer.succeed(None)
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, [1, 2, 3], _reactor=r)
    self.assertEqual(0, exitError.code)
    self.assertEqual(args, [1, 2, 3])