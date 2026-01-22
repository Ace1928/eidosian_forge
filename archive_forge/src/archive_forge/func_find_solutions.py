import re, sys, os, tempfile, json
def find_solutions(ideal, doubles=1):
    assert doubles in [1, 2, 4]
    prec = 53 * doubles
    tmpdir = tempfile.mkdtemp()
    infile = tmpdir + os.sep + 'polys.txt'
    outfile = tmpdir + os.sep + 'out.txt'
    ideal_to_file(ideal, tmpdir + os.sep + 'polys.txt')
    flag = {1: '-b', 2: '-b2', 4: '-b4'}[doubles]
    os.system('phc ' + flag + ' ' + infile + ' ' + outfile)
    ans = parse_file(outfile, prec)
    os.system('rm -rf tmpdir')
    return ans