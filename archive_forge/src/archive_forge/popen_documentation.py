import random
import subprocess
import sys
from fixtures import Fixture
Create a PopenFixture

        :param get_info: Optional callback to control the behaviour of the
            created process. This callback takes a kwargs dict for the Popen
            call, and should return a dict with any desired attributes.
            Only parameters that are supplied to the Popen call are in the
            dict, making it possible to detect the difference between 'passed
            with a default value' and 'not passed at all'.

            e.g. 
            def get_info(proc_args):
                self.assertEqual(subprocess.PIPE, proc_args['stdin'])
                return {'stdin': StringIO('foobar')}

            The default behaviour if no get_info is supplied is for the return
            process to have returncode of None, empty streams and a random pid.

            After communicate() or wait() are called on the process object,
            the returncode is set to whatever get_info returns (or 0 if
            get_info is not supplied or doesn't return a dict with an explicit
            'returncode' key).
        