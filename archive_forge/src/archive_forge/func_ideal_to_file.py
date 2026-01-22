import re, sys, os, tempfile, json
def ideal_to_file(ideal, filename):
    outfile = open(filename, 'w')
    polys = ideal.gens()
    outfile.write('%d\n' % len(polys))
    for p in polys:
        outfile.write('   ' + remove_forbidden(repr(p)) + ';\n')
    outfile.close()