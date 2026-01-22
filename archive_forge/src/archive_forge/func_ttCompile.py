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
@Timer(log, 'Done compiling TTX in %(time).3f seconds')
def ttCompile(input, output, options):
    input_name = input
    if input == '-':
        input, input_name = (sys.stdin, sys.stdin.name)
    output_name = output
    if output == '-':
        output, output_name = (sys.stdout.buffer, sys.stdout.name)
    log.info('Compiling "%s" to "%s"...' % (input_name, output))
    if options.useZopfli:
        from fontTools.ttLib import sfnt
        sfnt.USE_ZOPFLI = True
    ttf = TTFont(options.mergeFile, flavor=options.flavor, recalcBBoxes=options.recalcBBoxes, recalcTimestamp=options.recalcTimestamp)
    ttf.importXML(input)
    if options.recalcTimestamp is None and 'head' in ttf and (input is not sys.stdin):
        mtime = os.path.getmtime(input)
        ttf['head'].modified = timestampSinceEpoch(mtime)
    ttf.save(output)