from prov.model import ProvDocument, Namespace, Literal, PROV, Identifier
import datetime
def w3c_publication_2():
    ex = Namespace('ex', 'http://example.org/')
    rec = Namespace('rec', 'http://example.org/record')
    w3 = Namespace('w3', 'http://www.w3.org/TR/2011/')
    hg = Namespace('hg', 'http://dvcs.w3.org/hg/prov/raw-file/9628aaff6e20/model/releases/WD-prov-dm-20111215/')
    g = ProvDocument()
    g.entity(hg['Overview.html'], {'prov:type': 'file in hg'})
    g.entity(w3['WD-prov-dm-20111215'], {'prov:type': 'html4'})
    g.activity(ex['rcp'], None, None, {'prov:type': 'copy directory'})
    g.wasGeneratedBy('w3:WD-prov-dm-20111215', 'ex:rcp', identifier=rec['g'])
    g.entity('ex:req3', {'prov:type': Identifier('http://www.w3.org/2005/08/01-transitions.html#pubreq')})
    g.used('ex:rcp', 'hg:Overview.html', identifier='rec:u')
    g.used('ex:rcp', 'ex:req3')
    g.wasDerivedFrom('w3:WD-prov-dm-20111215', 'hg:Overview.html', 'ex:rcp', 'rec:g', 'rec:u')
    g.agent('ex:webmaster', {'prov:type': 'Person'})
    g.wasAssociatedWith('ex:rcp', 'ex:webmaster')
    return g