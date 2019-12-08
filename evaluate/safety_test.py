import pandas as pd
import numpy as np
from scipy import stats

from evaluate.pdis_evaluate import PDISEvaluate


class SafetyTest:

    def __init__(self, safety_df: pd.DataFrame, num_state_variables: int, num_actions: int, num_features_size: int, gamma=1.0):
        self._safety_df = safety_df
        self._num_states_variables = num_state_variables
        self._num_actions = num_actions
        self._num_features_size = num_features_size
        self._gamma = gamma

    def __call__(self, theta_c, theta_b, safety_val: float, confidence=0.9):
        pdis_eval = PDISEvaluate(num_state_variables=self._num_states_variables, num_actions=self._num_actions,
                                 num_features_size=self._num_features_size, gamma=self._gamma)
        pdis_h = []
        for idx, row in self._safety_df.iterrows():
            pdis_h.append(pdis_eval(row, theta_c, theta_b))
        pdis_h = np.array(pdis_h)
        n = self._safety_df.shape[0]
        pdis_d = np.mean(pdis_h)
        std_d = np.sqrt(np.sum(np.square(pdis_h - pdis_d)) / (n - 1))
        t_val = stats.t.ppf(confidence, n-1)
        if pdis_d - std_d*t_val/np.sqrt(n) > safety_val:
            return True

        return False
