from llvmlite import binding as llvm
def create_pass_manager_builder(opt=2, loop_vectorize=False, slp_vectorize=False):
    """
    Create an LLVM pass manager with the desired optimisation level and options.
    """
    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = opt
    pmb.loop_vectorize = loop_vectorize
    pmb.slp_vectorize = slp_vectorize
    pmb.inlining_threshold = _inlining_threshold(opt)
    return pmb