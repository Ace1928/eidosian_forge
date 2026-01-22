import enum
from pyomo.opt.results.container import MapContainer, ScalarType
@staticmethod
def to_solver_status(tc):
    """Maps a TerminationCondition to SolverStatus based on enum value

        Parameters
        ----------
        tc: TerminationCondition

        Returns
        -------
        SolverStatus
        """
    if tc in {TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations, TerminationCondition.minFunctionValue, TerminationCondition.minStepLength, TerminationCondition.globallyOptimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible, TerminationCondition.optimal, TerminationCondition.maxEvaluations, TerminationCondition.other}:
        return SolverStatus.ok
    if tc in {TerminationCondition.unbounded, TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded, TerminationCondition.invalidProblem, TerminationCondition.intermediateNonInteger, TerminationCondition.noSolution}:
        return SolverStatus.warning
    if tc in {TerminationCondition.solverFailure, TerminationCondition.internalSolverError, TerminationCondition.error}:
        return SolverStatus.error
    if tc in {TerminationCondition.userInterrupt, TerminationCondition.resourceInterrupt, TerminationCondition.licensingProblems}:
        return SolverStatus.aborted
    return SolverStatus.unknown