import os
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.fileutils import this_file_dir, PYOMO_ROOT_DIR
import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup
import pyomo.environ
def run_solver(self, *_args, **kwds):
    if self.solve:
        args = ['solve']
        args.append('--solver=' + self.solver)
        args.append('--save-results=result.yml')
        args.append('--results-format=yaml')
        args.append('--solver-options="lemke_start=automatic output_options=yes"')
    else:
        args = ['convert']
    args.append('-c')
    args.append('--symbolic-solver-labels')
    args.append('--file-determinism=2')
    if False:
        args.append('--stream-solver')
        args.append('--tempdir=' + currdir)
        args.append('--keepfiles')
        args.append('--logging=debug')
    args = args + list(_args)
    os.chdir(currdir)
    print('***')
    print(' '.join(args))
    try:
        output = pyomo_main.main(args)
    except SystemExit:
        output = None
    except:
        output = None
        raise
    cleanup()
    print('***')
    return output