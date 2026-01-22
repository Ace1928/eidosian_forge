import snappy
import regina
import snappy.snap.t3mlite as t3m
import snappy.snap.t3mlite.spun as spun
def hash_t3m_surface(surface):
    ans = [surface.EulerCharacteristic]
    ans += sorted(list(surface.EdgeWeights))
    ans += sorted(list(surface.Quadvector))
    return ans