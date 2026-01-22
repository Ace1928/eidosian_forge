import snappy
import regina
import snappy.snap.t3mlite as t3m
import snappy.snap.t3mlite.spun as spun
def to_regina(snappy_manifold):
    return regina.NTriangulation(snappy_manifold._to_string())