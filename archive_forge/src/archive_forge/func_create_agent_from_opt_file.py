from parlai.core.build_data import modelzoo_path
from parlai.core.loader import load_agent_module
from parlai.core.loader import register_agent  # noqa: F401
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
import copy
import os
import parlai.utils.logging as logging
def create_agent_from_opt_file(opt: Opt):
    """
    Load agent options and module from file if opt file exists.

    Checks to see if file exists opt['model_file'] + ".opt"; if so, load up the
    options from the file and use that to create an agent, loading the model
    type from that file and overriding any options specified in that file when
    instantiating the agent.

    If that file does not exist, return None.
    """
    model_file = opt['model_file']
    optfile = model_file + '.opt'
    if not os.path.isfile(optfile):
        return None
    opt_from_file = Opt.load(optfile)
    for arg in NOCOPY_ARGS:
        if arg in opt_from_file:
            del opt_from_file[arg]
    if opt.get('override'):
        for k, v in opt['override'].items():
            if k in opt_from_file and str(v) != str(opt_from_file.get(k)):
                logging.warn(f'Overriding opt["{k}"] to {v} (previously: {opt_from_file.get(k)})')
            opt_from_file[k] = v
    model_class = load_agent_module(opt_from_file['model'])
    if hasattr(model_class, 'upgrade_opt'):
        opt_from_file = model_class.upgrade_opt(opt_from_file)
    for k, v in opt.items():
        if k not in opt_from_file:
            opt_from_file[k] = v
    opt_from_file['model_file'] = model_file
    if not opt_from_file.get('dict_file'):
        opt_from_file['dict_file'] = model_file + '.dict'
    elif opt_from_file.get('dict_file') and (not os.path.isfile(opt_from_file['dict_file'])):
        old_dict_file = opt_from_file['dict_file']
        opt_from_file['dict_file'] = model_file + '.dict'
    if not os.path.isfile(opt_from_file['dict_file']):
        warn_once('WARNING: Neither the specified dict file ({}) nor the `model_file`.dict file ({}) exists, check to make sure either is correct. This may manifest as a shape mismatch later on.'.format(old_dict_file, opt_from_file['dict_file']))
    compare_init_model_opts(opt, opt_from_file)
    return model_class(opt_from_file)