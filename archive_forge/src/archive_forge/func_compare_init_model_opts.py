from parlai.core.build_data import modelzoo_path
from parlai.core.loader import load_agent_module
from parlai.core.loader import register_agent  # noqa: F401
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
import copy
import os
import parlai.utils.logging as logging
def compare_init_model_opts(opt: Opt, curr_opt: Opt):
    """
    Print loud warning when `init_model` opts differ from previous configuration.
    """
    if opt.get('init_model') is None:
        return
    opt['init_model'] = modelzoo_path(opt['datapath'], opt['init_model'])
    optfile = opt['init_model'] + '.opt'
    if not os.path.isfile(optfile):
        return
    init_model_opt = Opt.load(optfile)
    extra_opts = {}
    different_opts = {}
    exempt_opts = ['model_file', 'dict_file', 'override', 'starttime', 'init_model', 'batchindex']
    for k, v in init_model_opt.items():
        if k not in exempt_opts and k in init_model_opt and (init_model_opt[k] != curr_opt.get(k)):
            if isinstance(v, list):
                if init_model_opt[k] != list(curr_opt[k]):
                    different_opts[k] = ','.join([str(x) for x in v])
            else:
                different_opts[k] = v
    for k, v in curr_opt.items():
        if k not in exempt_opts and k not in init_model_opt:
            if isinstance(v, list):
                extra_opts[k] = ','.join([str(x) for x in v])
            else:
                extra_opts[k] = v
    extra_strs = ['{}: {}'.format(k, v) for k, v in extra_opts.items()]
    if extra_strs:
        logging.warn('your model is being loaded with opts that do not exist in the model you are initializing the weights with: {}'.format(','.join(extra_strs)))
    different_strs = ['--{} {}'.format(k.replace('_', '-'), v) for k, v in different_opts.items()]
    if different_strs:
        logging.warn('your model is being loaded with opts that differ from the model you are initializing the weights with. Add the following args to your run command to change this: \n{}'.format(' '.join(different_strs)))