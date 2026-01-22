from fontTools.cffLib import maxStackLimit
def generalizeProgram(program, getNumRegions=None, **kwargs):
    return commandsToProgram(generalizeCommands(programToCommands(program, getNumRegions), **kwargs))