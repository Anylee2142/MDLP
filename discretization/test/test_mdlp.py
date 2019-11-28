from discretization import ent, MDLP

import unittest

import numpy as np
import pandas as pd
from pandas.util import testing as pdt
import multiprocessing as mp

from sklearn.datasets import load_iris

from scipy.stats import entropy

class TestMDLP(unittest.TestCase):
    # TODO: root path <-> system path

    def test_custom_entropy(self):
        freq = pd.Series(data=[0,0,1,1,1,0,1], name='freq')

        event_freq = freq.value_counts()
        P = event_freq / event_freq.sum()

        self.assertGreaterEqual(ent(freq, base=2), entropy(P, base=2))
        self.assertGreaterEqual(ent(freq, base=np.e), entropy(P, base=np.e))

        self.assertAlmostEqual(ent(freq, base=2), entropy(P, base=2))
        self.assertAlmostEqual(ent(freq, base=np.e), entropy(P, base=np.e))

    def test_original_should_not_be_changed(self):
        data = load_iris()
        features = pd.DataFrame(data['data'], columns=['a', 'b', 'c', 'd'])
        target = pd.Series(data=data['target'], name='target')
        features_copy = features.copy()

        mdlp = MDLP(con_features=list(features.columns), base=2, max_cutpoints=3)
        mdlp.fit_transform(X=features, y=target)

        pdt.assert_frame_equal(features_copy, features)

    def test_procs_task(self):
        con_features = [str(each) for each in range(33)]

        mdlp = MDLP(con_features=con_features, base=2, max_cutpoints=3, n_jobs=8)
        print(mdlp.procs_task)
        self.assertEqual(8, len(mdlp.procs_task.keys()))
        self.assertEqual(8, mdlp.num_of_procs)

        mdlp = MDLP(con_features=con_features, base=2, max_cutpoints=3, n_jobs=15)
        self.assertEqual(mp.cpu_count(), mdlp.num_of_procs)

        mdlp = MDLP(con_features=con_features, base=2, max_cutpoints=3, n_jobs=-20)
        self.assertEqual(1, mdlp.num_of_procs)

    def test_parallel_iris_small_features(self):
        data = load_iris()
        features = pd.DataFrame(data['data'], columns=['a', 'b', 'c', 'd'])
        target = pd.Series(data=data['target'], name='target')

        single_mdlp = MDLP(con_features=features.columns, base=2, max_cutpoints=3)
        single_dis = single_mdlp.fit_transform(X=features, y=target)

        parallel_mdlp4 = MDLP(con_features=features.columns, base=2, max_cutpoints=3, n_jobs=4)
        parallel_dis4 = parallel_mdlp4.fit_transform(X=features, y=target)

        parallel_mdlp10 = MDLP(con_features=features.columns, base=2, max_cutpoints=3, n_jobs=10)
        parallel_dis10 = parallel_mdlp10.fit_transform(X=features, y=target)

        self.assertEqual(single_mdlp.cutpoints, parallel_mdlp4.cutpoints)
        self.assertEqual(parallel_mdlp4.cutpoints, parallel_mdlp10.cutpoints)
        pdt.assert_frame_equal(single_dis, parallel_dis4)
        pdt.assert_frame_equal(parallel_dis4, parallel_dis10)

if __name__ == '__main__':
    unittest.main()