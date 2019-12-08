import itertools
import math

import numpy as np
import pandas as pd

from policies.tabular_softmax import TabularSoftmax


class PDISEvaluate:

    def __init__(self, num_state_variables: int, num_actions: int, num_features_size: int, gamma=1.0):
        self._num_state_variables = num_state_variables
        self._num_actions = num_actions
        self._num_features_size = num_features_size
        self._state_feature_vectors = np.asarray(
            list(itertools.product(np.arange(self._num_features_size), repeat=self._num_state_variables)))
        self._gamma = gamma

    def __call__(self, history: np.array, theta_e, theta_b):
        pi_e = TabularSoftmax(numStates=1, numActions=self._num_actions)
        pi_b = TabularSoftmax(numStates=1, numActions=self._num_actions)
        cum_wt = 1.0
        pdis = 0

        i = 0
        while i < len(history) and not math.isnan(history[i]):
            s_t = history[i:i+self._num_state_variables]
            a_t = history[i + self._num_state_variables].astype(int)
            r_t = history[i + self._num_state_variables + 1]
            phi_state = np.cos(np.pi * self._state_feature_vectors.dot(s_t))
            theta_state_e = theta_e.reshape(phi_state.shape[0], -1).T.dot(phi_state)
            theta_state_b = theta_b.reshape(phi_state.shape[0], -1).T.dot(phi_state)
            pi_e.parameters = theta_state_e
            pi_b.parameters = theta_state_b
            cum_wt *= pi_e(0, a_t)/pi_b(0, a_t)
            pdis += cum_wt*r_t
            i += self._num_state_variables + 2

        # print(pdis)
        return pdis

