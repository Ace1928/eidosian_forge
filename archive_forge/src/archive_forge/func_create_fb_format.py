import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile
def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    with open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        with open(inpath) as f:
            lines = [line.strip('\n') for line in f]
            for i in range(len(lines)):
                use = True
                if dtype == 'train' and i % 20 == 0:
                    use = False
                if dtype == 'valid' and i % 20 != 0:
                    use = False
                if use:
                    xy = lines[i].split('OUT: ')
                    x = xy[0].split('IN: ')[1].rstrip(' ').lstrip(' ')
                    y = xy[1].rstrip(' ').lstrip(' ')
                    s = '1 ' + x + '\t' + y
                    fout.write(s + '\n\n')