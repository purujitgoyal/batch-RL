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
        self._candidate_df = candidate_df.to_numpy()
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
        while not cma_es.stop():
            X = cma_es.ask()
            cma_es.tell(X, self.run_multiprocessing(self.calculate_pdis, X, len(X)))

            cma_es.logger.add()
            cma_es.disp()

        if cma_es.result[1] == sys.maxsize:
            print("No Policy Found")
            return self._theta_b, None

        return cma_es.result[0], cma_es.result[1]

    def calculate_pdis(self, theta_c):
        # pdis_h = []
        # for idx, row in self._candidate_df.iterrows():
        #     pdis_h.append(self.pdis_eval(row, theta_c, self._theta_b))

        # pdis_h = self._candidate_df.apply(lambda row: self.pdis_eval(row, theta_c, self._theta_b), axis=1)
        # pdis_h = np.apply_along_axis(lambda row: self.pdis_eval(row, theta_c, self._theta_b), axis=1,
                                     # arr=self._candidate_df)
        pdis_h = self.pdis_eval(self._candidate_df, theta_c, self._theta_b)

        # pdis_h = np.array(pdis_h)
        n = self._candidate_df.shape[0]

        pdis_d = np.mean(pdis_h)
        print(pdis_d)
        std_d = np.sqrt(np.sum(np.square(pdis_h - pdis_d)) / (n - 1))
        t_val = stats.t.ppf(self._confidence, self._safe_df_size - 1)
        # print(2*std_d * t_val / np.sqrt(self._safe_df_size))
        if pdis_d - 2*std_d * t_val / np.sqrt(self._safe_df_size) > self._safety_val:
            return -pdis_d

        return sys.maxsize

    @staticmethod
    def run_multiprocessing(func, i, n_processors=8):
        with Pool(processes=n_processors) as pool:
            return pool.map(func, i)
