import copy, logging
from pyomo.common.dependencies import numpy
def solve_tear_wegstein(self, G, order, function, tears, outEdges, iterLim, tol, tol_type, report_diffs, accel_min, accel_max):
    """
        Use Wegstein to solve tears. If multiple tears are given
        they are solved simultaneously.

        Arguments
        ---------
            order
                List of lists of order in which to calculate nodes
            tears
                List of tear edge indexes
            iterLim
                Limit on the number of iterations to run
            tol
                Tolerance at which iteration can be stopped
            accel_min
                Minimum value for Wegstein acceleration factor
            accel_max
                Maximum value for Wegstein acceleration factor
            tol_type
                Type of tolerance value, either "abs" (absolute) or
                "rel" (relative to current value)

        Returns
        -------
            list
                List of lists of diff history, differences between input and
                output values at each iteration
        """
    hist = []
    if not len(tears):
        self.run_order(G, order, function, tears)
        return hist
    logger.info('Starting Wegstein tear convergence')
    itercount = 0
    ignore = tears + outEdges
    gofx = self.generate_gofx(G, tears)
    x = self.generate_first_x(G, tears)
    err = self.compute_err(gofx, x, tol_type)
    hist.append(err)
    if report_diffs:
        print('Diff matrix:\n%s' % err)
    if numpy.max(numpy.abs(err)) < tol:
        logger.info('Wegstein converged in %s iterations' % itercount)
        return hist
    x_prev = x
    gofx_prev = gofx
    x = gofx
    self.pass_tear_wegstein(G, tears, gofx)
    while True:
        itercount += 1
        logger.info('Running Wegstein iteration %s' % itercount)
        self.run_order(G, order, function, ignore)
        gofx = self.generate_gofx(G, tears)
        err = self.compute_err(gofx, x, tol_type)
        hist.append(err)
        if report_diffs:
            print('Diff matrix:\n%s' % err)
        if numpy.max(numpy.abs(err)) < tol:
            break
        if itercount > iterLim:
            logger.warning('Wegstein failed to converge in %s iterations' % iterLim)
            return hist
        denom = x - x_prev
        old_settings = numpy.seterr(divide='ignore', invalid='ignore')
        slope = numpy.divide(gofx - gofx_prev, denom)
        numpy.seterr(**old_settings)
        slope[numpy.isnan(slope)] = 0
        slope[numpy.isinf(slope)] = 0
        accel = slope / (slope - 1)
        accel[accel < accel_min] = accel_min
        accel[accel > accel_max] = accel_max
        x_prev = x
        gofx_prev = gofx
        x = accel * x_prev + (1 - accel) * gofx_prev
        self.pass_tear_wegstein(G, tears, x)
    self.pass_edges(G, outEdges)
    logger.info('Wegstein converged in %s iterations' % itercount)
    return hist