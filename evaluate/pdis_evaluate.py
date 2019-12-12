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
        # pi_e = TabularSoftmax(numStates=1, numActions=self._num_actions)
        # pi_b = TabularSoftmax(numStates=1, numActions=self._num_actions)
        # cum_wt = 1.0
        # pdis = 0

        # history = history[~np.isnan(history)]
        a_t = history[:, np.arange(start=self._num_state_variables, stop=history.shape[1], step=2+self._num_state_variables)].astype(int)
        history = np.delete(history, np.arange(start=self._num_state_variables, stop=history.shape[1], step=2+self._num_state_variables), axis=1)
        r_t = history[:, np.arange(start=self._num_state_variables, stop=history.shape[1], step=1+self._num_state_variables)]
        history = np.delete(history, np.arange(start=self._num_state_variables, stop=history.shape[1], step=1+self._num_state_variables), axis=1)

        # i = 0
        # while i < len(history) and not math.isnan(history[i]):
        #     s_t = history[i:i+self._num_state_variables]
        #     a_t = history[i + self._num_state_variables].astype(int)
        #     r_t = history[i + self._num_state_variables + 1]
        #     phi_state = np.cos(np.pi * self._state_feature_vectors.dot(s_t))
        #     # print(phi_state)
        #     # theta_state_e = theta_e.reshape(phi_state.shape[0], -1).T.dot(phi_state)
        #     theta_state_e = theta_e.reshape(self._num_actions, -1).dot(phi_state)
        #     theta_state_b = theta_b.reshape(self._num_actions, -1).dot(phi_state)
        #     # theta_state_b = theta_b.reshape(phi_state.shape[0], -1).T.dot(phi_state)
        #     pi_e.parameters = theta_state_e
        #     pi_b.parameters = theta_state_b
        #     cum_wt *= pi_e(0, a_t)/pi_b(0, a_t)
        #     pdis += cum_wt*r_t
        #     i += self._num_state_variables + 2

        phi_state = np.cos(np.pi*self._state_feature_vectors.dot(history.reshape(history.shape[0], 1, -1)))
        phi_state = np.transpose(phi_state, (1, 0, 2))
        theta_state_e = theta_e.reshape(self._num_actions, -1).dot(phi_state)
        theta_state_b = theta_b.reshape(self._num_actions, -1).dot(phi_state)
        theta_state_e = np.transpose(theta_state_e, (1, 2, 0))
        theta_state_b = np.transpose(theta_state_b, (1, 2, 0))
        # print(theta_state_b.shape)
        pi_e = self._get_softmax_probabilities(theta_state_e, a_t)
        pi_b = self._get_softmax_probabilities(theta_state_b, a_t)
        pi_ratio = np.cumprod(pi_e, axis=1)/np.cumprod(pi_b, axis=1)
        pdis = np.sum(pi_ratio*r_t, axis=1)
        # print(pdis)
        return pdis

    @staticmethod
    def _get_softmax_probabilities(theta_state, a_t):
        theta_state -= np.max(theta_state, axis=2, keepdims=True)
        theta_state = np.exp(theta_state)
        theta_state /= np.sum(theta_state, axis=2, keepdims=True)
        # temp[:, [0, 1, 2, 3, 4], a][np.arange(10), np.arange(10), :]
        action_prob = np.zeros((theta_state.shape[0], theta_state.shape[1]))
        for i in range(action_prob.shape[0]):
            action_prob[i] = theta_state[i][np.arange(theta_state.shape[1]), a_t[i]]
        return action_prob


# array([[ 0.13688028, -1.33540445, -0.69409199, -0.45440682, -1.80099557],
#        [-0.60329296, -1.24087111, -0.6864155 , -1.64648225, -0.10026528],
#        [ 1.7869418 ,  0.83949602, -1.74174719, -1.5766203 , -1.60758748],
#        [-1.27500073, -0.3858875 ,  0.71383273, -0.7783492 , -1.05891522],
#        [-1.16657052, -0.25443323,  1.9246645 , -0.75419615, -0.73346504],
#        [-0.00436847,  1.27100562,  1.16848581,  1.38763054, -0.12549326],
#        [ 0.25480272,  0.71223267, -2.72081754, -0.59450274,  0.85132434],
#        [-0.80039544, -0.2035634 ,  0.35397146,  0.12362152,  0.53735029],
#        [-1.88419538, -0.51777634,  0.27942546,  0.4520592 , -0.01860036],
#        [ 0.54561706,  2.29881392,  0.3872141 , -1.42233991,  0.45319337]])
