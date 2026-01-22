def generate_subsolver_results(self, include_local=True, include_global=True):
    """
        Generate flattened sequence all Pyomo SolverResults objects
        for all ``SeparationSolveCallResults`` objects listed in
        the local and global ``SeparationLoopResults``
        attributes of `self`.

        Yields
        ------
        pyomo.opt.SolverResults
        """
    if include_local and self.local_separation_loop_results is not None:
        all_local_call_results = self.local_separation_loop_results.solver_call_results.values()
        for solve_call_res in all_local_call_results:
            for res in solve_call_res.results_list:
                yield res
    if include_global and self.global_separation_loop_results is not None:
        all_global_call_results = self.global_separation_loop_results.solver_call_results.values()
        for solve_call_res in all_global_call_results:
            for res in solve_call_res.results_list:
                yield res