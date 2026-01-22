from fontTools.ttLib import TTFont, TTLibError
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.unicode import setUnicodeData
from fontTools.misc.textTools import Tag, tostr
from fontTools.misc.timeTools import timestampSinceEpoch
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
import os
import sys
import getopt
import re
import logging
@Timer(log, 'Done dumping TTX in %(time).3f seconds')
def ttDump(input, output, options):
    input_name = input
    if input == '-':
        input, input_name = (sys.stdin.buffer, sys.stdin.name)
    output_name = output
    if output == '-':
        output, output_name = (sys.stdout, sys.stdout.name)
    log.info('Dumping "%s" to "%s"...', input_name, output_name)
    if options.unicodedata:
        setUnicodeData(options.unicodedata)
    ttf = TTFont(input, 0, ignoreDecompileErrors=options.ignoreDecompileErrors, fontNumber=options.fontNumber)
    ttf.saveXML(output, tables=options.onlyTables, skipTables=options.skipTables, splitTables=options.splitTables, splitGlyphs=options.splitGlyphs, disassembleInstructions=options.disassembleInstructions, bitmapGlyphDataFormat=options.bitmapGlyphDataFormat, newlinestr=options.newlinestr)
    ttf.close()