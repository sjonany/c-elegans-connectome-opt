"""
Helper methods to generate V(t) data and their derivations
"""
import numpy as np
from sklearn.decomposition import PCA

import connectomes
import project_path
from model.neural_model import NeuralModel

def get_jimin_3neuron_dynamics(simul_ts, dt):
  """
  Get the oscillatory dynamics produced by Jimin's 3 neurons setup.
  """
  Gg, Gs, is_inhibitory = connectomes.get_jimin_3neuron_connectome()
  I_ext = get_jimin_3neuron_Iext()
  N = 3
  return run_neural_model(N, Gg, Gs, is_inhibitory, I_ext, simul_ts, dt)

def get_jimin_3neuron_Iext():
  return 100000 * np.array([0, 0.03 ,0])

def run_neural_model(N, Gg, Gs, is_inhibitory, I_ext, simul_ts, dt):
  """
  Generate V(t) data for the given connectome and stimulation.
  """
  model = NeuralModel(
    N = N,
    Gg = Gg,
    Gs = Gs,
    is_inhibitory = is_inhibitory,
    I_ext = I_ext)
  model.dt = dt
  
  (v_mat, s_mat, v_normalized_mat) = model.run(simul_ts)
  return v_normalized_mat

def get_top_mode(v_mat):
  """
  Get just the first PCA component of a population voltage trace
  """
  pca = PCA(n_components = 1)
  projected_X = pca.fit_transform(v_mat)
  return projected_X[:,0]
