"""
Optimizer based on Ke li et al., 2016's Learning to optimize,
BUT, instead of using Guided Policy Search, we just use the simpler
policy gradient > REINFORCE method implemented by Morvan Zhou.
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py
"""

import pdb
import numpy as np
import policy_gradient_reinforce as policy_gradient
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from enum import IntEnum

# Morvan's code is against TF 1.0
tf.disable_v2_behavior() 

# Each policy can only increase, decrease, or stay by a certain ACTION_DELTA.
NUM_ACTION = 3

# TODO: Tweak all the parameters below per optimization problem
ACTION_DELTA = 0.1
# How many timesteps to look back for state. Ke li 2017 used 25.
NUM_TIMESTEPS_FOR_STATE = 10
# The number of episodes / full game. Each episode starts from the initial state w0.
NUM_EPISODES = 50
# Number of steps per episode
EPISODE_LEN = 15
# Learning rate for policy gradient
LEARNING_RATE = 0.02
# Highest verbosity level = FINE. 0 = no debugging information
class VerbosityLevel(IntEnum):
  NONE = 1
  INFO = 2
  FINE = 3

VERBOSITY_LEVEL = VerbosityLevel.NONE

def optimize_with_w_and_dfw(f, w0):
  # Gradient-less state.
  no_op_gradient = lambda w: 0.0
  return optimize_with_rl(f, no_op_gradient, w0, state_creator_w_and_dfw)

def state_creator_dfw_and_gradient(w_histories, f_histories, g_histories, last_f_wt):
  """
  # State space is as described by Ke li 2017:
  # - Changes in the objective value at the current location relative to the objective value
  #    at the ith most recent location for all i
  # - Gradient of the objective function evaluated at the ith most recent location for all i 
  # We do NOT include the current weights as state like Ke li 2017:
  # "The current location is only used to compute the cost; because the policy
  # "should not depend on the absolute coordinates of the current location,
  # "we exclude it from the input that is fed into the neural net"
  # The format is: delta_f * NUM_TIMESTEPS_FOR_STATE, then grad_f * NUM_TIMESTEPS_FOR_STATE
  #   Most recent chunk is at the front of the array
  """ 
  xt_fs = last_f_wt - np.array(f_histories)
  xt_gs = np.array(g_histories).flatten()
  return np.append(xt_fs, xt_gs)

def state_creator_gradient(w_histories, f_histories, g_histories, last_f_wt):
  return np.array(g_histories).flatten()

def state_creator_w_and_dfw(w_histories, f_histories, g_histories, last_f_wt):
  """
  State space where we have the past params and the f values
  """ 
  xt_ws = np.array(w_histories).flatten()
  xt_fs = last_f_wt - np.array(f_histories)
  return np.append(xt_ws, xt_fs)

def state_creator_dfw(w_histories, f_histories, g_histories, last_f_wt):
  """
  State space like Ke li, but without the gradient information
  """ 
  return last_f_wt - np.array(f_histories)

def state_creator_fw(w_histories, f_histories, g_histories, last_f_wt):
  return np.array([last_f_wt] + f_histories[:-1])

def optimize_with_rl(f, g, w0, state_creator):
  """
  Params
    f - The objective function to be minimized. f(w) should return a scalar.
    g - The gradient function.
    w0 - Initial w.
  Returns
    w such that f(w) is minimized
  """
  w0 = np.copy(w0)
  tf.reset_default_graph()
  opt_param_dim = len(w0)

  # TODO: Use scipy to get x0. Right now everything is just set to w0, f(w0)
  f_w0 = f(w0)
  g_w0 = g(w0)
  
  # wt and f_histories are not explicitly encoded as states, but are necessary to be updated
  # because they are used for next state calculations.
  wt = w0
  
  # Store histories of w, f(w(t)) and g(w(t)). First element is the most recent.
  w_histories = [np.copy(w0)] * NUM_TIMESTEPS_FOR_STATE
  f_histories = [f_w0] * NUM_TIMESTEPS_FOR_STATE
  g_histories = [g_w0] * NUM_TIMESTEPS_FOR_STATE

  # Initially, we don't have any changes in obj functions
  x0 = state_creator(w_histories, f_histories, g_histories, f_w0)
  xt = x0
  
  state_dim = len(x0)
  # Each policies[i] is an RL model for tweaking the 1 parameter dimension w[i]
  policies = []
  for i in range(opt_param_dim):
    RL = policy_gradient.PolicyGradient(
      name_suffix=str(i),
      n_actions=NUM_ACTION,
      n_features=state_dim,
      learning_rate=LEARNING_RATE,
      reward_decay=1.0
    )
    policies.append(RL)
  
  # Total reward from start to finish of each episode
  ep_rewards = []
  ep_fws = []
  best_wt = None
  min_fwt = None
  for ep in range(NUM_EPISODES):
    for t in range(EPISODE_LEN):
      # Every policy will have its own reward.
      rs = np.zeros(opt_param_dim)
      last_f_wt = 0
      for i in range(opt_param_dim):
        # The actions are numbers from 0 to NUM_ACTION - 1
        action = policies[i].choose_action(xt)
        if VERBOSITY_LEVEL >= VerbosityLevel.FINE:
          print("t=%d i=%d action=%s" % (t, i, action))
          print("wt_prev=%s" % (wt))
        wt[i] += convert_action_to_dw(action)
        last_f_wt = f(wt)
        
        if min_fwt is None or min_fwt > last_f_wt:
          best_wt = wt
          min_fwt = last_f_wt

        last_g_wt = g(wt)
        rs[i] = -last_f_wt
        if VERBOSITY_LEVEL >= VerbosityLevel.FINE:
          print("wt_next=%s" % (wt))
          print("f(wt)=%s" % last_f_wt)
          print("rs=%s" % (rs))
        # TODO: Should we make the state observed by each policy change?
        # Right now this means that every time an agent acts,
        # the other agents are part of the stochastic environment,
        # And yet my reward is only computed off my immediate action.
        # Ez change: Just do the xt updates here.
        policies[i].store_transition(xt, action, rs[i])
      if VERBOSITY_LEVEL >= VerbosityLevel.FINE:
        print("xt_prev=%s" % (xt))
      xt = state_creator(w_histories, f_histories, g_histories, last_f_wt)
      
      # Rotate histories w/ the most recent f_wt entry
      # Rotate f_histories w/ the most recent f_wt entry
      f_histories = [last_f_wt] + f_histories[:-1]
      g_histories = [last_g_wt] + g_histories[:-1]
      w_histories = [np.copy(wt)] + w_histories[:-1]
      if VERBOSITY_LEVEL >= VerbosityLevel.FINE:
        print("xt_next=%s" % (xt))
        print("f_hist=%s" % (f_histories))
        print("g_hist=%s" % (g_histories))
        print("w_hist=%s" % (w_histories))
    # The end of 1 episode  
    for policy in policies:
      ep_rewards.append(sum(policy.ep_rs))
      if VERBOSITY_LEVEL >= VerbosityLevel.FINE:
        print("Sum rl.ep_rs = %.2f" % ep_rewards[-1])
      vt = policy.learn()
      if VERBOSITY_LEVEL >= VerbosityLevel.FINE:
        print("rl.learn.vt = %s" % vt)
        plt.plot(vt)    # plot the episode vt
        plt.xlabel('episode steps')
        plt.ylabel('normalized state-action value')
        plt.show()
    ep_fws.append(last_f_wt)
    if VERBOSITY_LEVEL >= VerbosityLevel.FINE:
      print("Episode %d, f(w) = %.2f, w = %s" % (ep+1, last_f_wt, wt))

  if VERBOSITY_LEVEL >= VerbosityLevel.INFO:
    plt.plot(ep_rewards)
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.show()

    plt.plot(ep_fws)
    plt.xlabel('episode')
    plt.ylabel('last f(w)')
    plt.show()
  return best_wt
  
def convert_action_to_dw(action):
  if action == 0:
    return 0
  elif action == 1:
    return -ACTION_DELTA
  else:
    return ACTION_DELTA