import os
import pyomo.common.unittest as unittest
def test_baselines(self):
    filenames = ['simple_odbc', 'diet']
    basePath = os.path.split(os.path.abspath(__file__))[0]
    for fn in filenames:
        iniPath = os.path.join(basePath, 'baselines', '{0}.ini'.format(fn))
        outPath = os.path.join(basePath, 'baselines', '{0}.out'.format(fn))
        config = ODBCConfig(filename=iniPath)
        config.write(outPath)
        written = ODBCConfig(filename=outPath)
        self.assertEqual(config, written)
        try:
            os.remove(outPath)
        except:
            pass