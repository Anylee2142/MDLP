import numpy as np
import modin.pandas as pd
from typing import Tuple, List, Dict
from .discretizer import Discretizer

def log_(v, base=2) -> float:
    return np.log(v) / np.log(base)

def ent(vector: pd.Series, base=2) -> float:
    '''
    Compute entropy from frequencies, not probability
    :param vector: pd.Series that holds frequencies
    :param base: base number for entropy
    :return: entropy
    '''
    events = vector.value_counts()
    pi = events / events.sum()

    return -np.sum(log_(v=pi, base=base) * pi)

def get_subset(feature: pd.Series, T: float, target: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series] or None:
    '''
    Divides S into S1, S2 according to T
    :param feature: continuous column
    :param T: threshold that divides S1, S2
    :param target: target column
    :return: S1, S2 according to T
    '''
    if len(feature) != len(target):
        # 'lengths are not same between feature, target'
        return None
    if len(feature) == 0:
        # 'feature series should not be 0 length'
        return None

    feature_name = feature.name
    S = pd.concat([feature, target], axis=1)
    S1 = S[S[feature_name] < T]
    S2 = S[S[feature_name] > T]

    if len(S1) == 0 or len(S2) == 0:
        return None

    if len(S1) + len(S2) != len(target[S1.index]) + len(target[S2.index]):
        # 'lengths are not same between S1, S2'
        return None

    return S[target.name], target[S1.index], target[S2.index]

def E(feature: pd.Series, T: float, target: pd.Series, base=2) -> np.inf or float:
    '''
    Entropy after dividing set S according to threshold T
    :param feature: continuous column
    :param T: threshold that divides S1, S2
    :param target: target column
    :return: class entropy according to T
    '''
    result = get_subset(feature, T, target)

    if result is None:
        # Because entropy behaves same as cost.
        # if something wrong, let it not considered through making it infinity
        return np.inf

    S_target, S1_target, S2_target = result

    return (len(S1_target) * ent(S1_target, base=base) + len(S2_target) * ent(S2_target, base=base)) / (
                len(S1_target) + len(S2_target))

def Gain(feature: pd.Series, T: float, target: pd.Series, base=2) -> float:
    '''
    Information gain
    :param feature: continuous column
    :param T: threshold that divides S1, S2
    :param target: target column
    :return: information gain according to T
    '''
    return ent(target, base=base) - E(feature, T, target)

def delta(feature: pd.Series, T: float, target: pd.Series, base=2) -> float:
    '''
    :param feature: continuous column
    :param T: threshold that divides S1, S2
    :param target: target column
    :return: refer to paper
    '''
    result = get_subset(feature, T, target)

    if result is None:
        return np.inf

    S_target, S1_target, S2_target = result

    k = len(S_target.unique())
    k1 = len(S1_target.unique())
    k2 = len(S2_target.unique())

    return (np.log2(3 ** k - 2) - (
            k * ent(S_target, base=base) - k1 * ent(S1_target, base=base) - k2 * ent(S2_target, base=base))) / (
                   len(S1_target) + len(S2_target))

def N_1(feature, base=2) -> float:
    N = len(feature)
    if N == 0:
        return np.inf
    return log_(v=N-1, base=base) / N

class MDLP(Discretizer):
    def __init__(self, con_features: List[str], base=2, max_cutpoints=3, n_jobs=1, is_modin=True):
        assert base > 0, 'base of logarithm should be bigger than 0'
        assert base != 1, 'base of logarithm should not be 1'
        assert isinstance(con_features, list), '`con_features` should be list'

        super().__init__(
            con_features=con_features,
            max_cutpoints=max_cutpoints,
            n_jobs=n_jobs,
            is_modin=is_modin
        )

        # Discretizer.__init__(
        #     self=self,
        #     con_features=con_features,
        #     max_cutpoints=max_cutpoints,
        #     n_jobs=n_jobs,
        #     is_modin=is_modin
        # )

        self.base = base

    def _find_best_cutpoint(self, feature: pd.Series, possible_cutpoints: list, target: pd.Series) -> float:
        '''
        Find best cutpoint that minimize E and satisfy mdlp criterion
        :param feature: continuous column
        :param possible_cutpoints: every possible cutpoint
        :param target: target column
        :return:
            1. best_cutpoint = best threshold (cutpoint) minimizing E, and satisfy mdlp condition
            2. first_subrange = S1 or S2 that has more samples. None if no more search
            3. second_subrange = S1 or S2 that has less samples. None if no more search
        '''
        if len(feature) == 0 or len(target) == 0:
            return None, None, None

        best_cutpoint = -1
        best_entropy = np.inf

        n_1 = N_1(feature, base=self.base)

        for T in possible_cutpoints:
            # TODO: implement below with pd.apply or map
            curr_entropy = E(feature, T, target)
            # To be cutpoint,
            # 1. it should minimize class entropy
            if best_entropy > curr_entropy:
                best_entropy = curr_entropy
                best_cutpoint = T

        # 2. and pass mdlp criterion
        if Gain(feature, best_cutpoint, target, self.base) <= n_1 + delta(feature, best_cutpoint, target, self.base):
            return None, None ,None

        result = get_subset(feature, best_cutpoint, target)

        if result is None:
            return None, None, None

        _, S1_target, S2_target = result
        S1_feature = feature[S1_target.index]
        S2_feature = feature[S2_target.index]

        first_subrange = (S1_feature, S1_target) if len(S1_target) > len(S2_target) else (S2_feature, S2_target)
        second_subrange = (S2_feature, S2_target) if len(S1_target) > len(S2_target) else (S1_feature, S1_target)

        return best_cutpoint, first_subrange, second_subrange

    def _get_possible_cutpoints(self, feature: pd.Series) -> List[float]:
        '''
        Get boundaries from given feature
        :param feature: continuous feature
        :return: possible cutpoints
        '''
        uniques = feature.unique()
        mid_cutpoints = []
        for i in range(len(uniques[:-1])):
            mid_point = (uniques[i] + uniques[i + 1]) / 2
            mid_cutpoints.append(mid_point)
        return mid_cutpoints

    def _get_cutpoints(self, feature: pd.Series, target: pd.Series) -> List[float]:
        '''
        get mdlp cutpoints from one continuous feature
        :param feature: continuous column to be mdlp-discretized
        :param target: target column considered with mdlp
        :return: List that mdlp-cutpoints
        '''
        cutpoints = []
        sorted_feature = feature.sort_values()
        sorted_target = target[sorted_feature.index]
        subranges = [(sorted_feature, sorted_target)]

        num_iter = self.max_cutpoints

        while num_iter > 0 and subranges:
            feature, target = subranges.pop(0)
            # possible cutpoint should be middle point of adjacent two values, not itself
            possible_cutpoints = self._get_possible_cutpoints(feature)

            best_cutpoint, first_subrange, second_subrange = self._find_best_cutpoint(feature, possible_cutpoints, target)

            cutpoints.append(best_cutpoint)
            if first_subrange is not None and second_subrange is not None:
                subranges.append(first_subrange)
                subranges.append(second_subrange)
            num_iter -= 1

        cutpoints = list(filter(lambda elem: elem != None, cutpoints))
        cutpoints.sort()

        return cutpoints