"""Augmenter for ADMG. Only uses the Tian factorization.
"""
import numpy as np
import pandas as pd
from .augmenter_kernel import AugmenterKernel
from .vanilla import VanillaProductKernel, VanillaProductKernelConfig, KernelWeightComputer, UnconditionalWeightComputer

# Type hinting
from typing import Tuple, Iterable, Optional, List
from ananke.graphs import ADMG
from pandas import DataFrame as DF
from tqdm import tqdm

# from . import Timer

CONTI_OR_DISC = {
    'int64': 'u',
    'int32': 'u',
    'float64': 'c',
    'float32': 'c',
}


# @Timer.set(lambda t: print('[Timer] full_augmentation finished: ', t.time))
def _full_augmented_data(data_aug: pd.DataFrame,
                         kernels: Iterable[AugmenterKernel],
                         ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute the full augmentation.

    Parameters:
      data : the base data to use for the augmentation (in the current implementation, this has to be the same data set to which the kernels have been fit).
      kernels : vector-valued functions $f(c)$ that returns $(\tilde{p}(x_i | c))_i$ where $\tilde{p}(x_i | c)$ is the weighting of selecting $x_i$ (column $x$ of the $i$-th row of ``data``).
      log_weight_threshold : the lower log-weight threshold for performing the augmentation.

    Returns:
        Tuple containing

        - df : the data frame containing the augmented data. This can have duplicate values with the original data. The augmentation candidates whose weight is lower than the threshold are pruned.
        - weight : the relative weight of each augmented data computed from the kernels. The values may not sum to one because of the pruning (the values are not normalized after the pruning).
    """
    # @Timer.set(lambda t: print('[Timer] _one_step took ', t.time))
    def _one_step(current_df: pd.DataFrame, current_logweight: np.ndarray, kernel: AugmenterKernel) -> Tuple[pd.DataFrame, np.ndarray]:
        """Perform one step of the augmentation, corresponding to one depth of the probability tree.

        Parameters:
            current_df : The current augmented data.
            current_logweight : The current array of log-weights for the augmented data.
            data_ref : Reference data to be used for augmenting the data (typically the training data).
            kernel : The augmentation kernel to be used for computing the weights of the augmented data.

        Returns:
            Tuple containing

            - ``current_df`` : Augmented data.
            - ``current_logweight`` : Log weights of the augmented instances.
        """
        c = current_df[kernel.c_names]
        z = current_df[kernel.v_names[0]]
        w = kernel.c_weighter(np.array(z), np.array(c))  # (len_current_df, n)
        # Allow log(0) to be -inf.
        with np.errstate(divide='ignore'):
            logweights = np.log(w)
       
        assert np.sum(np.isnan(logweights)) == 0
        assert np.sum(logweights > 0) == 0
        _current_logweight_base = current_logweight + logweights
        _current_logweight = _current_logweight_base

        return _current_logweight

    # Initialize buffers.
    current_logweight = np.log(np.ones((len(data_aug), 1)))
    for kernel in kernels:
        print("kernel", kernel)
        current_logweight = _one_step(data_aug, current_logweight, kernel)
        if len(data_aug) == 0:
            return np.empty(0)
    # print(current_logweight)
    current_weight = np.exp(current_logweight)
    if np.sum(np.isnan(current_weight)) != 0:
        current_weight = np.nan_to_num(current_weight)
    return current_weight




class ADMGTianAugmenter:
    """Base class for the proposed augmentation method based on the Tian factorization (topological ADMG factorization)."""
    def __init__(self, graph: ADMG, top_order: Optional[List[str]] = None):
        """Constructor.

        Parameters:
            graph : the ADMG model to be used for the data augmentation.
            top_order : a valid topological order on the graph (a list of the vertex names).
                        If ``None`` is provided, it is automatically computed from the graph (default: ``None``).
        """
        self.graph = graph
        if top_order is None:
            top_order = self.graph.topological_sort()
        self.graph_top_order = top_order

    def prepare(self, data_ref: DF, data_aug: DF, weight_kernel_cfg: dict):
        """Perform Tian factorization and fit weight functions for the conditioning variables.

        Parameters:
            data_ref : Original Data for reference.
            data_aug : Data to be augmented.
            weight_kernel_cfg : Configuration of the kernels.
        """
        # Prepare dict (column name -> data type)
        dtypes = data_ref.dtypes.apply(lambda x: x.name).apply(
            lambda x: CONTI_OR_DISC[x]).to_dict()

        # Prepare factorization kernels
        factorization_kernels = []
        for v in tqdm(self.graph_top_order):
            #get mp data from data_ref
            v_data = np.array(data_ref[v])
            v_type = dtypes[v]
            mp = list(self.graph.markov_pillow([v], self.graph_top_order))
            mp_data = np.array(data_ref[mp])
            var_types = ''.join([dtypes[var] for var in mp])
            if len(mp) > 0:
                if weight_kernel_cfg['type'] == 'vanilla_kernel':
                    config = VanillaProductKernelConfig()
                    if weight_kernel_cfg.get('const_bandwidth', False):
                        config.conti_bw_method = lambda _: weight_kernel_cfg[
                            'bandwidth_temperature']
                    config.conti_bw_temperature = weight_kernel_cfg.get(
                        'bandwidth_temperature', 1.)
                    config.conti_kertype = weight_kernel_cfg.get(
                        'conti_kertype', 'gaussian')
                    _kernel = VanillaProductKernel(z_ref=v_data,
                                                   ztype=v_type,
                                                   mp_ref=mp_data,
                                                   vartypes=var_types,
                                                   config=config)
                    weighter = KernelWeightComputer(_kernel)
            else:
                n = len(data_aug)
                weighter = UnconditionalWeightComputer(n)
            factorization_kernels.append(AugmenterKernel([v], mp, weighter))
            # print(factorization_kernels)
        self.factorization_kernels = factorization_kernels
        self.data_ref = data_ref
        self.data_aug = data_aug
    
    def augment(self) -> np.ndarray:
        """Perform Tian factorization and augment the data accordingly.

        Parameters:
            weight_threshold : the lower weight threshold for performing the augmentation.
            normalize_threshold_by_data_size : whether to normalize the threshold by the data size.
                                               (``True``: divide the threshold by the data size)

        Returns:
            Tuple containing

            - augmented_data : the fully augmented data.
            - weights : the instance weights based on the kernel values.
        """

        self.data_aug = self.data_aug[self.data_ref.columns]
        self.data_aug = self.data_aug.astype(self.data_ref.dtypes)
        weights = _full_augmented_data(
            self.data_aug, self.factorization_kernels)
        # Reorder DataFrame columns
        return weights
    
if __name__ == "__main__":
    vertices = ["X1", "X2", "X3"]
    di_edges = [("X1", "X2"), ("X1", "X3")]
    bi_edges = []
    admg = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
    X = np.random.normal(0, 10, size=10)
    data_ref = pd.DataFrame({
        "X1": X,
        "X2": X + np.random.normal(0, 1, size=10),
        "X3": X + np.random.normal(0, 1, size=10),
    })

    # Augmentation dataset (10 samples)
    data_aug = pd.DataFrame({
        "X1": X + np.random.normal(0, 1, size=10),
        "X2": X + 2 * np.random.normal(0, 1, size=10),
        "X3": X + 2 * np.random.normal(0, 1, size=10),
    })

    augmenter_config =  {
            'type': 'vanilla_kernel',

            'conti_kertype': 'gaussian', # type of kernel to use for a continuous variable
            'conti_bw_method': 'normal_reference',
            'conti_bw_temperature': 1,

            'dis_kertype': 'indicator', # type of kernel to use for a discrete and ordered variable
            'dis_bw_method': 'indicator',

            'const_bandwidth': False, # False: adapt the bandwidth to the type of kerel used
            'bandwidth_temperature': 0.001 # bandwidth used in case of constant bandwidth
        }

    augmenter = ADMGTianAugmenter(graph=admg)  # G is from ananke
    augmenter.prepare(data_ref, data_aug, augmenter_config)
    weights = augmenter.augment()

    print("Augmented weights:", weights)
    


    
