import numpy as np
from scipy import integrate, linalg, sparse
import pdb
import time

class NeuralModel:
  """
  A model of C-elegans V(t) dynamics
  Code adapted from https://github.com/shlizee/C-elegans-Neural-Interactome/blob/master/initialize.py
  """

  def __init__(self, N, Gg, Gs, is_inhibitory, I_ext):
    """
    N - number of neurons
    Gg - NxN. gap junction, should be symmetric.
    Gs - NxN. synaptic junction, can be asymmetric.
    is_inhibitory - Nx1. whether or not neuron i is inhibitory.
    I_ext - Nx1. current injection to each neuron.
    """
    self.N = N
    self.Gg = Gg
    self.Gs = Gs
    self.E = -48.0 * is_inhibitory
    self.I_ext = I_ext

    # Or else we have uncomparable results across optimization runs
    np.random.seed(0)
    self.init_conds = 10**(-4)*np.random.normal(0, 0.94, 2*self.N)

    self.dt = 0.01

    # Cell membrane capacitance. Default: 1.5 pF / 100 = 0.015 arb (arbitrary unit)
    self.C = 0.015

    # Cell membrane conductance, for calculating I_leak. Default: 10pS / 100 = 0.1 arb
    self.Gc = 0.1

    # Leakage potential (mV)
    self.Ec = -35.0

    # ggap Default: 100pS / 100 = 1 arb
    self.ggap = 1.0

    # gsyn Default: 100pS / 100 = 1 arb
    self.gsyn = 1.0

    # Synaptic activity
    # Synaptic activity's rise time
    self.ar = 1.0/1.5
    # Synaptic activity's decay time 
    self.ad = 5.0/1.5
    # Width of the sigmoid (mv^-1)
    self.B = 0.125

    self.compute_Vth()

  def compute_Vth(self):
    """
    Vth computation that I wrote from scratch, and that matches my math derivations more.
    Validations:
    - Ran interactome code and printed out their Vth.
      Vth 1 30 100, sum = -18.3342063908 -6.2498993778 -3.61729185518 -1194.25458719
    - compute_Vth(), this method, produces very similar results as well.
      Vth 1 30 100, sum = -18.334206390780512 -6.249899377800608 -3.6172918551786215 -1194.2545871854845
    - L2 norm of compute_Vth() and interactome's:  3.907985046680551e-14
    """
    b1 = -np.tile(self.Gc * self.Ec, self.N)
    # Interactome rounded to 4, so we followed suit.
    s_eq = round(self.ar / (self.ar + 2 * self.ad), 4)
    b3 = -s_eq * (self.Gs @ self.E)

    m1 = -self.Gc * np.identity(self.N)
    # N x 1, where each item is a row sum
    Gg_row_sums = self.Gg.sum(axis = 1)
    # m2 is a diagonal matrix with the negative row sums as the values
    m2 = - np.diag(Gg_row_sums)
    Gs_row_sums =  self.Gs.sum(axis = 1)
    m3 = - s_eq * np.diag(Gs_row_sums)
    # I think paper is missing m4. It shouldn't be the case that A is a completely diagonal matrix.
    # However, interactome github code seems to have done this correctly.
    # Our implementation is mathematically equivalent to the github code.
    m4 = self.Gg

    A = m1 + m2 + m3 + m4
    # b = b1 + b3
    b = b1 + b3 - self.I_ext
    self.A = A
    self.Vth = np.reshape(linalg.solve(A, b), self.N)

  def dynamic(self, t, state_vars):
    """Dictates the dynamics of the system.
    """
    v_arr, s_arr = np.split(state_vars, 2)

    # I_leak
    I_leak = self.Gc * (v_arr - self.Ec)

    # I_gap = sum_j G_ij (V_i - V_j) = V_i sum_j G_ij - sum_j G_ij V_j
    # The first term is a point-wise multiplication of V and G's squashed column.
    # The second term is matrix multiplication of G and V
    I_gap = self.Gg.sum(axis = 1) * v_arr - self.Gg @ v_arr
    
    # I_syn = sum_j G_ij s_j (V_i - E_j) = V_i sum_j G_ij s_j - sum_j G_j s_j E_j
    # First term is a point-wise multiplication of V and (Matrix mult of G and s)
    # Second term is matrix mult of G and (point mult of s_j and E_j)
    I_syn = v_arr * (self.Gs @ s_arr) - self.Gs @ (s_arr * self.E)

    dV = (-I_leak - I_gap - I_syn + self.I_ext) / self.C
    phi = np.reciprocal(1.0 + np.exp(-self.B*(v_arr - self.Vth)))
    syn_rise = self.ar * phi * (1 - s_arr)
                          
    syn_drop = self.ad * s_arr
    dS = syn_rise - syn_drop
    return np.concatenate((dV, dS))

  def get_normalized_v_arr(self, v_arr):
    """The paper performs analysis on this normalized v_arr.
    """
    vth_adjusted = v_arr - self.Vth
    vmax = 500

    # tanh: Similar to sigmoid, but squashes to between -1 and 1.
    # So, below readjusts value to range from -500 to 500.
    return vmax * np.tanh(np.divide(vth_adjusted, vmax)) 

  def run(self, num_timesteps):
    """Create initial conditions, then simulate dynamics num_timesteps times.
    Args:
      num_timesteps (int): The number of simulation timesteps to run for. Each timestep is dt = 0.01 second long.
    Returns:
      v_mat (num_timesteps x N): Each column is a voltage timeseries of a neuron. 
      s_mat (num_timesteps x N): Each column is an activation timeseries of a neuron's synaptic current.
      v_normalized_mat (num_timesteps x N): v_mat, but normalized just like the exported dynamics file from Interactome.
        Note that interactome's exported data starts from timestep 50 onwards, so make sure to truncate first 50 if you
        want to compare.
    """

    N = self.N
    dt = self.dt

    # The variables to store our complete timeseries data.
    v_mat = []
    s_mat = []
    v_normalized_mat = []

    dyn = integrate.ode(self.dynamic).set_integrator('vode', atol = 1e-3,
        min_step = dt*1e-6, method = 'bdf', with_jacobian = True,
        max_step = dt)
    dyn.set_initial_value(self.init_conds, 0)

    start_time = time.time()
    for t in range(num_timesteps):
      dyn.integrate(dyn.t + dt)
      v_arr = dyn.y[:N]
      s_arr = dyn.y[N:]
      v_normalized_arr = self.get_normalized_v_arr(v_arr)
      v_mat.append(v_arr)
      s_mat.append(s_arr)
      v_normalized_mat.append(v_normalized_arr)
    print("Total runtime = %.2fs" % (time.time() - start_time))
    return np.array(v_mat), np.array(s_mat), np.array(v_normalized_mat)