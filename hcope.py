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
    safety_val = 2

    # test_df = pd.read_csv('test_histories.csv', header=None)
    test_df = pd.read_csv('histories.csv', header=None).fillna(0)

    i = 0
    while i < 100:
        msk = np.random.rand(len(test_df)) < 0.6
        candidate_df = test_df[msk]
        safety_df = test_df[~msk]
        cs = CandidateSelection(candidate_df=candidate_df, safety_df_size=len(test_df),
                                num_state_variables=num_state_variables,
                                num_actions=num_actions, num_features_size=num_features_size + 1, theta_b=theta_b,
                                safety_val=safety_val, confidence=0.99)

        # cs.calculate_pdis(theta_b)
        # print(i)
        theta_c, pdis_val = cs(sigma=1, num_iter=50)
        print(theta_c)
        print(pdis_val)
        if pdis_val is None:
            continue
        #
        # # theta_c = [3.967075,2.961444,2.233000,-2.064813,-3.034357,2.489803,-4.278893,11.016205]
        # # theta_c = np.array(theta_c)
        #
        st = SafetyTest(safety_df, num_state_variables=num_state_variables, num_actions=num_actions,
                        num_features_size=num_features_size + 1)

        safe = st(theta_c, theta_b, safety_val=safety_val, confidence=0.99)
        if safe:
            np.savetxt(str(i) + ".csv", [theta_c], delimiter=",", fmt='%f')
            i += 1


    # for i in range(10):
    #     t = np.random.randn(i)
    #     np.savetxt(str(i)+".csv", [t], delimiter=",", fmt='%f')
#
