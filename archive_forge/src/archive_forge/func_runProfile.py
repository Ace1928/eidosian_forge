import numpy
def runProfile(command):
    import random
    random.seed(23)
    import profile
    import pstats
    datFile = '%s.prof.dat' % command
    profile.run('%s()' % command, datFile)
    stats = pstats.Stats(datFile)
    stats.strip_dirs()
    stats.sort_stats('time').print_stats()