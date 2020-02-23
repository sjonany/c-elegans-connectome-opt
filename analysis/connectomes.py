"""
Utility functions for generating or manipulating connectomes
"""
import numpy as np
import numpy.ma as ma
import pdb

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
  compact_vec = np.random.rand(num_gg_compact(N) + num_gs_compact(N))
  Gg, Gs = compact_to_model_param(compact_vec, N)
  # TODO: Discuss with team what fixed is_inhibitory do we want. For now I'll just hardcode to Jimin's
  is_inhibitory = np.array([1, 0 ,0]) # np.array([1] * N)
  return (Gg, Gs, is_inhibitory)

def compact_to_model_param(compact_vec, N):
  """
  Convert the compact vector representation of the model parameters, which will be used by optimization library,
  to the model representation (Gg, Gs) matrices
  The compact vector representation is composed of compact forms of Gg, then Gs.
  Compact form of Gg are just the lower triangle, because self edges are not allowed (zero diagonals), and upper
  triangle is just transpose. There is (N*N - N) / 2 of these. 
  Compact form of Gs are everything but the zero diagonals. There is (N*N - N) of these.
  """
  num_gg_items = num_gg_compact(N)
  Gg_compact = compact_vec[:num_gg_items]
  Gs_compact = compact_vec[num_gg_items:]

  Gg = compact_to_model_Gg(Gg_compact, N)
  Gs = compact_to_model_Gs(Gs_compact, N)

  return Gg, Gs

def model_to_compact_param(Gg, Gs, N):
  compact_vec = np.concatenate([
    model_to_compact_Gg(Gg, N),
    model_to_compact_Gs(Gs, N)
  ])
  return compact_vec

def model_to_compact_Gg(Gg, N):
  return Gg[get_lower_tri_mask(N)]

def model_to_compact_Gs(Gs, N):
  return np.concatenate([Gs[get_lower_tri_mask(N)], Gs[get_upper_tri_mask(N)]])

def compact_to_model_Gg(Gg_compact, N):
  Gg = np.zeros((N,N))
  fill_lower_tri(Gg, Gg_compact)
  # Copy the lower triangle to upper triangle, keeping in mind non-lower triangle values are zeroes.
  Gg += Gg.T
  return Gg

def compact_to_model_Gs(Gs_compact, N):
  gs_compact_n = num_gs_compact(N)
  lower_tri_n = int(gs_compact_n / 2)
  Gs = np.zeros((N,N))
  fill_lower_tri(Gs, Gs_compact[:lower_tri_n])
  fill_upper_tri(Gs, Gs_compact[lower_tri_n:])
  return Gs

def fill_lower_tri(mat, a):
  """
  Fill the lower triangle values of a matrix with values from array.
  Obtained from https://stackoverflow.com/a/51439529
  """
  n = mat.shape[0]
  mask = get_lower_tri_mask(n)
  mat[mask] = a

def get_lower_tri_mask(n):
  return np.tri(n,dtype=bool, k=-1)

def fill_upper_tri(mat, a):
  """
  Fill the upper triangle values of a matrix with values from array.
  See fill_lower_tri
  """
  n = mat.shape[0]
  mask = get_upper_tri_mask(n)
  mat[mask] = a

def get_upper_tri_mask(n):
  return np.logical_not(np.tri(n,dtype=bool, k=0))

def num_gg_compact(N):
  """
  Compact form of Gg are just the lower triangle, because self edges are not allowed (zero diagonals), and upper
  triangle is just transpose. There is (N*N - N) / 2 of these. 
  """
  return int((N*N - N) / 2)

def num_gs_compact(N):
  """
  Compact form of Gs are everything but the zero diagonals. There is (N*N - N) of these.
  """
  return int(N*N - N)
