import logging
import platform
from pyomo.common import Library
from pyomo.common.deprecation import deprecated
@deprecated('Use of get_gsl is deprecated and NO LONGER FUNCTIONS as of February 9, 2023. ', version='6.5.0')
def get_gsl(downloader):
    logger.info('As of February 9, 2023, AMPL GSL can no longer be downloaded        through download-extensions. Visit https://portal.ampl.com/        to download the AMPL GSL binaries.')