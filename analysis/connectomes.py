"""
Utility functions for generating or manipulating connectomes
"""
import numpy as np

def get_jimin_3neuron_connectome():
  """
  Return Gg (NxN), Gs (NxN), is_inhibitory (Nx1)
  This is the configuration used by Jimin, where we see stable oscillation with just 3 neurons.
  """
  Gg = np.array([[0, 8, 5],
                 [8, 0, 2],
                 [5, 2, 0]])
  Gs = np.array([[0, 2, 8],
                 [7, 0, 3],
                 [7, 7, 0]])
  is_inhibitory = np.array([1, 0 ,0])
  return (Gg, Gs, is_inhibitory)

def get_random_connectome(N):
  """
  Return Gg (NxN), Gs (NxN), is_inhibitory (Nx1)
  This is the configuration used by Jimin, where we see stable oscillation with just 3 neurons.
  """
  # TODO: Make Gg symmetric
  Gg = np.random.rand(N,N)
  Gs = np.random.rand(N,N)
  # TODO: Discuss with team what fixed is_inhibitory do we want. For now I'll just hardcode to Jimin's
  is_inhibitory = np.array([1, 0 ,0]) # np.array([1] * N)
  return (Gg, Gs, is_inhibitory)
