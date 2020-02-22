"""
Different distance metrics for comparing two types of dynamics
"""
import numpy as np
import dynamics

def preprocess_pop_dyn(pop_dyn, eval_ts):
	"""
	Preprocess population V(t) for evaluation.
	All the other methods in this file assumes they only have to compare two time series
	of equal length'
	Param
	- pop_dyn. (timesteps x N) matrix. The V(t)'s of all neurons
	- eval_ts. Number of timesteps at the end of the dynamics to compare against.
		Used for cropping. 
	"""

	# Crop for the interesting timestamps, then do PCA.
	# We don't want the transients to skew the PCA.
	cropped_pop_dyn = pop_dyn[-eval_ts:,:]
	return dynamics.get_top_mode(cropped_pop_dyn)

def ts_distance_euclidean(ts1, ts2):
  """
  Euclidean distance for two timeseries.
  """
  return np.linalg.norm(ts1 - ts2)