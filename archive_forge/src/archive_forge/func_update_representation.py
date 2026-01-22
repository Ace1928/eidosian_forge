import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
def update_representation(self, model):
    """Update model Representation"""
    self.model = model
    self.prefix = model.prefix
    self.dtype = model.dtype
    self.nobs = model.nobs
    self.k_endog = model.k_endog
    self.k_states = model.k_states
    self.k_posdef = model.k_posdef
    self.time_invariant = model.time_invariant
    self.endog = model.endog
    self.design = model._design.copy()
    self.obs_intercept = model._obs_intercept.copy()
    self.obs_cov = model._obs_cov.copy()
    self.transition = model._transition.copy()
    self.state_intercept = model._state_intercept.copy()
    self.selection = model._selection.copy()
    self.state_cov = model._state_cov.copy()
    self.missing = np.array(model._statespaces[self.prefix].missing, copy=True)
    self.nmissing = np.array(model._statespaces[self.prefix].nmissing, copy=True)
    self.shapes = dict(model.shapes)
    for name in self.shapes.keys():
        if name == 'obs':
            continue
        self.shapes[name] = getattr(self, name).shape
    self.shapes['obs'] = self.endog.shape
    self.initialization = model.initialization
    if model.initialization is not None:
        model._initialize_state()
        self.initial_state = np.array(model._statespaces[self.prefix].initial_state, copy=True)
        self.initial_state_cov = np.array(model._statespaces[self.prefix].initial_state_cov, copy=True)
        self.initial_diffuse_state_cov = np.array(model._statespaces[self.prefix].initial_diffuse_state_cov, copy=True)