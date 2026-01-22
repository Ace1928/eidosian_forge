import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def simple_recycle_model(self):
    m = ConcreteModel()
    m.comps = Set(initialize=['A', 'B', 'C'])
    m.feed = Block()
    m.feed.flow_out = Var(m.comps)
    m.feed.temperature_out = Var()
    m.feed.pressure_out = Var()
    m.feed.expr_var_idx_out = Var(m.comps)

    @m.feed.Expression(m.comps)
    def expr_idx_out(b, i):
        return -b.expr_var_idx_out[i]
    m.feed.expr_var_out = Var()
    m.feed.expr_out = -m.feed.expr_var_out

    @m.feed.Port()
    def outlet(b):
        return dict(flow=b.flow_out, temperature=b.temperature_out, pressure=b.pressure_out, expr_idx=b.expr_idx_out, expr=b.expr_out)

    def initialize_feed(self):
        pass
    m.feed.initialize = MethodType(initialize_feed, m.feed)
    m.mixer = Block()
    m.mixer.flow_in_side_1 = Var(m.comps)
    m.mixer.temperature_in_side_1 = Var()
    m.mixer.pressure_in_side_1 = Var()
    m.mixer.expr_var_idx_in_side_1 = Var(m.comps)

    @m.mixer.Expression(m.comps)
    def expr_idx_in_side_1(b, i):
        return -b.expr_var_idx_in_side_1[i]
    m.mixer.expr_var_in_side_1 = Var()
    m.mixer.expr_in_side_1 = -m.mixer.expr_var_in_side_1
    m.mixer.flow_in_side_2 = Var(m.comps)
    m.mixer.temperature_in_side_2 = Var()
    m.mixer.pressure_in_side_2 = Var()
    m.mixer.expr_var_idx_in_side_2 = Var(m.comps)

    @m.mixer.Expression(m.comps)
    def expr_idx_in_side_2(b, i):
        return -b.expr_var_idx_in_side_2[i]
    m.mixer.expr_var_in_side_2 = Var()
    m.mixer.expr_in_side_2 = -m.mixer.expr_var_in_side_2
    m.mixer.flow_out = Var(m.comps)
    m.mixer.temperature_out = Var()
    m.mixer.pressure_out = Var()
    m.mixer.expr_var_idx_out = Var(m.comps)

    @m.mixer.Expression(m.comps)
    def expr_idx_out(b, i):
        return -b.expr_var_idx_out[i]
    m.mixer.expr_var_out = Var()
    m.mixer.expr_out = -m.mixer.expr_var_out

    @m.mixer.Port()
    def inlet_side_1(b):
        return dict(flow=b.flow_in_side_1, temperature=b.temperature_in_side_1, pressure=b.pressure_in_side_1, expr_idx=b.expr_idx_in_side_1, expr=b.expr_in_side_1)

    @m.mixer.Port()
    def inlet_side_2(b):
        return dict(flow=b.flow_in_side_2, temperature=b.temperature_in_side_2, pressure=b.pressure_in_side_2, expr_idx=b.expr_idx_in_side_2, expr=b.expr_in_side_2)

    @m.mixer.Port()
    def outlet(b):
        return dict(flow=b.flow_out, temperature=b.temperature_out, pressure=b.pressure_out, expr_idx=b.expr_idx_out, expr=b.expr_out)

    def initialize_mixer(self):
        for i in self.flow_out:
            self.flow_out[i].value = value(self.flow_in_side_1[i] + self.flow_in_side_2[i])
        for i in self.expr_var_idx_out:
            self.expr_var_idx_out[i].value = value(self.expr_var_idx_in_side_1[i] + self.expr_var_idx_in_side_2[i])
        self.expr_var_out.value = value(self.expr_var_in_side_1 + self.expr_var_in_side_2)
        assert self.temperature_in_side_1.value == self.temperature_in_side_2.value
        self.temperature_out.value = value(self.temperature_in_side_1)
        assert self.pressure_in_side_1.value == self.pressure_in_side_2.value
        self.pressure_out.value = value(self.pressure_in_side_1)
    m.mixer.initialize = MethodType(initialize_mixer, m.mixer)
    m.unit = Block()
    m.unit.flow_in = Var(m.comps)
    m.unit.temperature_in = Var()
    m.unit.pressure_in = Var()
    m.unit.expr_var_idx_in = Var(m.comps)

    @m.unit.Expression(m.comps)
    def expr_idx_in(b, i):
        return -b.expr_var_idx_in[i]
    m.unit.expr_var_in = Var()
    m.unit.expr_in = -m.unit.expr_var_in
    m.unit.flow_out = Var(m.comps)
    m.unit.temperature_out = Var()
    m.unit.pressure_out = Var()
    m.unit.expr_var_idx_out = Var(m.comps)

    @m.unit.Expression(m.comps)
    def expr_idx_out(b, i):
        return -b.expr_var_idx_out[i]
    m.unit.expr_var_out = Var()
    m.unit.expr_out = -m.unit.expr_var_out

    @m.unit.Port()
    def inlet(b):
        return dict(flow=b.flow_in, temperature=b.temperature_in, pressure=b.pressure_in, expr_idx=b.expr_idx_in, expr=b.expr_in)

    @m.unit.Port()
    def outlet(b):
        return dict(flow=b.flow_out, temperature=b.temperature_out, pressure=b.pressure_out, expr_idx=b.expr_idx_out, expr=b.expr_out)

    def initialize_unit(self):
        for i in self.flow_out:
            self.flow_out[i].value = value(self.flow_in[i])
        for i in self.expr_var_idx_out:
            self.expr_var_idx_out[i].value = value(self.expr_var_idx_in[i])
        self.expr_var_out.value = value(self.expr_var_in)
        self.temperature_out.value = value(self.temperature_in)
        self.pressure_out.value = value(self.pressure_in)
    m.unit.initialize = MethodType(initialize_unit, m.unit)
    m.splitter = Block()

    @m.splitter.Block(m.comps)
    def flow_in(b, i):
        b.flow = Var()
    m.splitter.temperature_in = Var()
    m.splitter.pressure_in = Var()
    m.splitter.expr_var_idx_in = Var(m.comps)

    @m.splitter.Expression(m.comps)
    def expr_idx_in(b, i):
        return -b.expr_var_idx_in[i]
    m.splitter.expr_var_in = Var()
    m.splitter.expr_in = -m.splitter.expr_var_in
    m.splitter.flow_out_side_1 = Var(m.comps)
    m.splitter.temperature_out_side_1 = Var()
    m.splitter.pressure_out_side_1 = Var()
    m.splitter.expr_var_idx_out_side_1 = Var(m.comps)

    @m.splitter.Expression(m.comps)
    def expr_idx_out_side_1(b, i):
        return -b.expr_var_idx_out_side_1[i]
    m.splitter.expr_var_out_side_1 = Var()
    m.splitter.expr_out_side_1 = -m.splitter.expr_var_out_side_1
    m.splitter.flow_out_side_2 = Var(m.comps)
    m.splitter.temperature_out_side_2 = Var()
    m.splitter.pressure_out_side_2 = Var()
    m.splitter.expr_var_idx_out_side_2 = Var(m.comps)

    @m.splitter.Expression(m.comps)
    def expr_idx_out_side_2(b, i):
        return -b.expr_var_idx_out_side_2[i]
    m.splitter.expr_var_out_side_2 = Var()
    m.splitter.expr_out_side_2 = -m.splitter.expr_var_out_side_2

    @m.splitter.Port()
    def inlet(b):
        return dict(flow=Reference(b.flow_in[:].flow), temperature=b.temperature_in, pressure=b.pressure_in, expr_idx=b.expr_idx_in, expr=b.expr_in)

    @m.splitter.Port()
    def outlet_side_1(b):
        return dict(flow=b.flow_out_side_1, temperature=b.temperature_out_side_1, pressure=b.pressure_out_side_1, expr_idx=b.expr_idx_out_side_1, expr=b.expr_out_side_1)

    @m.splitter.Port()
    def outlet_side_2(b):
        return dict(flow=b.flow_out_side_2, temperature=b.temperature_out_side_2, pressure=b.pressure_out_side_2, expr_idx=b.expr_idx_out_side_2, expr=b.expr_out_side_2)

    def initialize_splitter(self):
        recycle = 0.1
        prod = 1 - recycle
        for i in self.flow_in:
            self.flow_out_side_1[i].value = prod * value(self.flow_in[i].flow)
            self.flow_out_side_2[i].value = recycle * value(self.flow_in[i].flow)
        for i in self.expr_var_idx_in:
            self.expr_var_idx_out_side_1[i].value = prod * value(self.expr_var_idx_in[i])
            self.expr_var_idx_out_side_2[i].value = recycle * value(self.expr_var_idx_in[i])
        self.expr_var_out_side_1.value = prod * value(self.expr_var_in)
        self.expr_var_out_side_2.value = recycle * value(self.expr_var_in)
        self.temperature_out_side_1.value = value(self.temperature_in)
        self.temperature_out_side_2.value = value(self.temperature_in)
        self.pressure_out_side_1.value = value(self.pressure_in)
        self.pressure_out_side_2.value = value(self.pressure_in)
    m.splitter.initialize = MethodType(initialize_splitter, m.splitter)
    m.prod = Block()
    m.prod.flow_in = Var(m.comps)
    m.prod.temperature_in = Var()
    m.prod.pressure_in = Var()
    m.prod.actual_var_idx_in = Var(m.comps)
    m.prod.actual_var_in = Var()

    @m.prod.Port()
    def inlet(b):
        return dict(flow=b.flow_in, temperature=b.temperature_in, pressure=b.pressure_in, expr_idx=b.actual_var_idx_in, expr=b.actual_var_in)

    def initialize_prod(self):
        pass
    m.prod.initialize = MethodType(initialize_prod, m.prod)

    @m.Arc(directed=True)
    def stream_feed_to_mixer(m):
        return (m.feed.outlet, m.mixer.inlet_side_1)

    @m.Arc(directed=True)
    def stream_mixer_to_unit(m):
        return (m.mixer.outlet, m.unit.inlet)

    @m.Arc(directed=True)
    def stream_unit_to_splitter(m):
        return (m.unit.outlet, m.splitter.inlet)

    @m.Arc(directed=True)
    def stream_splitter_to_mixer(m):
        return (m.splitter.outlet_side_2, m.mixer.inlet_side_2)

    @m.Arc(directed=True)
    def stream_splitter_to_prod(m):
        return (m.splitter.outlet_side_1, m.prod.inlet)
    TransformationFactory('network.expand_arcs').apply_to(m)
    m.feed.flow_out['A'].fix(100)
    m.feed.flow_out['B'].fix(200)
    m.feed.flow_out['C'].fix(300)
    m.feed.expr_var_idx_out['A'].fix(10)
    m.feed.expr_var_idx_out['B'].fix(20)
    m.feed.expr_var_idx_out['C'].fix(30)
    m.feed.expr_var_out.fix(40)
    m.feed.temperature_out.fix(450)
    m.feed.pressure_out.fix(128)
    return m