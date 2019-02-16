import numpy as np
import pandas as pd

from abc import *
from typing import Tuple, List, Dict, Any
import multiprocessing as mp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

def _queue_get(queue:mp.Queue) -> List[Dict[str, float]]:
    result = list()

    while True:
        elem = queue.get()
        if elem is None:
            break
        result.append(elem)

    return result

class Discretizer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    def __init__(self, con_features: List[str], max_cutpoints=3, n_jobs=1):
        self.con_features = con_features
        self.max_cutpoints = max_cutpoints
        self.cutpoints = dict()

        self.num_of_procs = None
        self.procs_task = self._assign_tasks(n_jobs)

    def _assign_tasks(self, n_jobs: int) -> Dict[int, List[str]]:
        '''
        Assign groups of features to each process
        :param n_jobs: how many processes to make
        :return: Dict that {process index: List of features to handle}
        '''
        def _check_cores(n_jobs: int) -> int:
            '''
            Determine how many processes to set
            :param n_jobs: how many processes to make
            :return:
            '''
            num_of_procs = 1
            upperbound = mp.cpu_count()

            if n_jobs > 0:
                num_of_procs = upperbound if n_jobs >= upperbound else n_jobs
            elif n_jobs < 0:
                num_of_procs = upperbound + n_jobs + 1 if n_jobs >= -upperbound else 1

            return num_of_procs

        self.num_of_procs = _check_cores(n_jobs)
        procs_task = {each:[] for each in range(self.num_of_procs)}

        for each in range(len(self.con_features)):
            proc_idx = each % self.num_of_procs
            proc_task = procs_task[proc_idx]
            proc_task.append(self.con_features[each])

            procs_task.update({
                proc_idx: proc_task
            })

        return procs_task
    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''
        Get mdlp cutpoints
        :param X: pd.DataFrame that features
        :param y: pd.Series that target
        :return: self
        '''
        self.cutpoints = self._prepare_and_get_cutpoints(features=X, target=y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Discretized each continuous feature according to mdlp cutpoints
        :param X: pd.DataFrame that features
        :return: discretized X
        '''
        Y = X.copy()
        for feature_name, cutpoints in self.cutpoints.items():
            if not cutpoints:
                continue
            # TODO: 1.traverse 2.apply 3. traverse
            Y[feature_name] = pd.Series(
                data=np.searchsorted(cutpoints, X.loc[:, feature_name]),
                name=feature_name)
        return Y

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        return self.fit(X, y).transform(X)

    @abstractmethod
    def _get_cutpoints(self):
        '''
        This method should describe how to get cutpoints according to each algorithm
        '''
        raise NotImplementedError

    def _get_cutpoints_multiple_features(self, features: List[pd.Series], target: pd.Series, queue=mp.Queue):
        '''
        Adapter function between get_cutpoints <-> mdlp_cutpoints, for parallel execution
        :param features:
        :param target:
        :param max_cutpoints:
        :param queue:
        :return: nothing
        '''
        queue.put({
            feature.name: self._get_cutpoints(
                feature=feature,
                target=target)
            for feature in features
        })

    def _prepare_and_get_cutpoints(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, List[float]]:
        '''
        Get mdlp cutpoints from every continuous feature
        :param features: pd.Dataframe that self.con_features
        :param target_: pd.Series that target
        :return: mdlp cutpoints for every continuous feature {feature name: List of cutpoints}
        '''

        # Validate if features are continuous
        for feature_name in self.con_features:
            try:
                features.loc[:, feature_name].astype(float)
            except ValueError as ve:
                print(ve)

        # LabelEncoding target
        target = pd.Series(
            data=LabelEncoder().fit_transform(target),
            name=target.name)

        cutpoints_each_feature = dict()

        if self.num_of_procs == 1:
            # TODO: process all features in one go, not individually
            cutpoints_each_feature.update({
                feature_name: self._get_cutpoints(
                        feature=features.loc[:, feature_name],
                        target=target,
                        ) for feature_name in self.con_features
            })
        else:
            procs = list()
            queue = mp.Queue()

            for proc_idx, proc_task in self.procs_task.items():

                if proc_task:
                    procs.append(mp.Process(
                        target=self._get_cutpoints_multiple_features,
                        kwargs={
                            'features': [features.loc[:, feature_name] for feature_name in proc_task],
                            'target': target,
                            'queue': queue
                        }
                    ))

            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join()

            queue.put(None)
            for each in _queue_get(queue):
                cutpoints_each_feature.update(each)

        return cutpoints_each_feature