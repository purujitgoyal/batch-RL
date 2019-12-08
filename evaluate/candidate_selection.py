import sys
from multiprocessing import Pool
from multiprocessing import freeze_support

import cma
import pandas as pd
import numpy as np
from scipy import stats

from evaluate.pdis_evaluate import PDISEvaluate


class CandidateSelection:

    def __init__(self, candidate_df: pd.DataFrame, safety_df_size: int, num_state_variables: int, num_actions: int,
                 num_features_size: int, theta_b, safety_val: float, confidence=0.9, gamma=1.0):
        self._candidate_df = candidate_df
        self._safe_df_size = safety_df_size
        self._num_states_variables = num_state_variables
        self._num_actions = num_actions
        self._num_features_size = num_features_size
        self._gamma = gamma
        self._theta_b = theta_b
        self._safety_val = safety_val
        self._confidence = confidence
        self.pdis_eval = PDISEvaluate(num_state_variables=self._num_states_variables, num_actions=self._num_actions,
                                      num_features_size=self._num_features_size, gamma=self._gamma)

    def __call__(self, sigma=2, num_iter=100, bounds=None):

        cma_es = cma.CMAEvolutionStrategy(self._theta_b, sigma)
        print(cma_es.popsize)
        if bounds is not None:
            cma_es = cma.CMAEvolutionStrategy(self._theta_b, sigma, {'bounds': bounds})

        i = 0
        # freeze_support()
        while i < num_iter:
            X = cma_es.ask()
            #     print(len(X))
            cma_es.tell(X, self.run_multiprocessing(self._calculate_pdis, X, len(X)))
            # cma_es.tell(X, [self._calculate_pdis(x) for x in X])
            cma_es.logger.add(modulo=10)
            cma_es.disp(modulo=100)
            print(cma_es.result[0])
            i += 1

        return cma_es.result[0], cma_es.result[1]

    def _calculate_pdis(self, theta_c):
        pdis_h = []
        for idx, row in self._candidate_df.iterrows():
            pdis_h.append(self.pdis_eval(row, theta_c, self._theta_b))

        # print(pdis_h)
        pdis_h = np.array(pdis_h)
        n = self._candidate_df.shape[0]
        # print(pdis_h)
        pdis_d = np.mean(pdis_h)
        print(pdis_d)
        std_d = np.sqrt(np.sum(np.square(pdis_h - pdis_d)) / (n - 1))
        t_val = stats.t.ppf(self._confidence, self._safe_df_size - 1)
        if pdis_d - std_d * t_val / np.sqrt(self._safe_df_size) > self._safety_val:
            return -pdis_d

        return 100000000

    @staticmethod
    def run_multiprocessing(func, i, n_processors=8):
        with Pool(processes=n_processors) as pool:
            return pool.map(func, i)
