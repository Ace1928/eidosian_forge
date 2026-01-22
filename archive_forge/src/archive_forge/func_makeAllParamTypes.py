from _paramtreecfg import cfg
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.parametertree.parameterTypes import GroupParameter
def makeAllParamTypes():
    children = []
    for name, paramCfg in cfg.items():
        if ' ' in name:
            children.append(makeMetaChild(name, paramCfg))
        else:
            children.append(makeChild(name, paramCfg))
    params = Parameter.create(name='Example Parameters', type='group', children=children)
    sliderGrp = params.child('Sample Slider')
    slider = sliderGrp.child('widget')
    slider.setOpts(limits=[0, 100])

    def setOpt(_param, _val):
        infoChild.setOpts(**{_param.name(): _val})
    meta = params.child('Applies to All Types')
    infoChild = meta.child('Extra Information')
    for child in meta.children()[1:]:
        child.sigValueChanged.connect(setOpt)

    def onChange(_param, _val):
        if _val == 'Use span':
            span = slider.opts.pop('span', None)
            slider.setOpts(span=span)
        else:
            limits = slider.opts.pop('limits', None)
            slider.setOpts(limits=limits)
    sliderGrp.child('How to Set').sigValueChanged.connect(onChange)

    def activate(action):
        for ch in params:
            if isinstance(ch, GroupParameter):
                ch.setOpts(expanded=action.name() == 'Expand All')
    for name in ('Collapse', 'Expand'):
        btn = Parameter.create(name=f'{name} All', type='action')
        btn.sigActivated.connect(activate)
        params.insertChild(0, btn)
    missing = [typ for typ in set(PARAM_TYPES).difference(_encounteredTypes) if not typ.startswith('_')]
    if missing:
        raise RuntimeError(f'{missing} parameters are not represented')
    return params