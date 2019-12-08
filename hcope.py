import numpy as np
import pandas as pd

from evaluate.candidate_selection import CandidateSelection
from evaluate.safety_test import SafetyTest

if __name__ == '__main__':
    num_state_variables = 1
    num_actions = 2
    num_features_size = 1
    theta_b = np.array([0.01, -0.01, 1, 1])
    # theta_b = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    n_episodes = 200000
    safety_val = 1.25

    # test_df = pd.read_csv('test_histories.csv', header=None)
    test_df = pd.read_csv('histories.csv', header=None)
    msk = np.random.rand(len(test_df)) < 0.4
    candidate_df = test_df[msk]
    safety_df = test_df[~msk]

    for i in range(100):
        cs = CandidateSelection(candidate_df=candidate_df, safety_df_size=len(safety_df),
                                num_state_variables=num_state_variables,
                                num_actions=num_actions, num_features_size=num_features_size + 1, theta_b=theta_b,
                                safety_val=safety_val, confidence=0.95)
        theta_c, pdis_val = cs(sigma=1, num_iter=50)
        print(theta_c)
        print(pdis_val)

        # theta_c = [0.33304397, -1.3536084, -0.391217, -2.93140976, -4.07265164, -2.13137546, -1.94398795, 2.53769193]
        # theta_c = np.array(theta_c)
        st = SafetyTest(safety_df, num_state_variables=num_state_variables, num_actions=num_actions,
                        num_features_size=num_features_size + 1)

        print(st(theta_c, theta_b, safety_val=safety_val, confidence=0.95))
        np.savetxt(str(i) + ".csv", [theta_c], delimiter=",", fmt='%f')
    #
    # for i in range(10):
    #     t = np.random.randn(i)
    #     np.savetxt(str(i)+".csv", [t], delimiter=",", fmt='%f')
#