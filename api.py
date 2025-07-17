"""Versions of the method (corresponding to different modeling choices, subroutines, etc.) will be defined in this layer. """
import numpy as np

# Type hinting
from typing import Iterable, Optional, Union, Tuple
import pandas as pd
from ananke.graphs import ADMG
from .method_config import *
from .augmenter_full import ADMGTianAugmenter


class EagerCausalDataAugmentation:
    """Implementation of Causal Data Augmentation.
    Augments the data and returns the augmented data (i.e., not lazy = eager).
    Suitable for those predictor classes that go better with
    one-time data augmentation than on-the-fly augmentation.
    """
    def __init__(self, method_config: AugmenterConfig = FullAugment()):
        """Constructor.

        Parameters:
            method_config : the config of the method.
        """
        self.validate_config(method_config)
        self.method_config = method_config

    def validate_config(self, method_config: AugmenterConfig) -> None:
        """Check the validity of the method config.

        Parameters:
            method_config : Method configuration to be validated.
        """
        pass

    def augment(self, data_ref: pd.DataFrame, data_aug: pd.DataFrame, estimated_graph: ADMG) -> np.ndarray:
        """Generate augmented data. Does not consider overlapping, etc., against the original data.

        Parameters:
            data: The source domain data to be used for fitting the novelty detector.
            estimated_graph: The ADMG object used for performing the augmentation.

        Returns:
            One of the following:

            - Tuple of ``(augmented_data, weights)`` : if ``self.sampling_method`` is ``'full'``.
            - augmented_data : if ``self.sampling_method`` is ``'stochastic'``.

        Examples:
            >> weight_threshold = 1e-5
            >> augmenter = EagerCausalDataAugmentation(FullAugment(weight_threshold))
            >> raise NotImplementedError()
        """
        if isinstance(self.method_config, FullAugment):
            full_augmenter = ADMGTianAugmenter(estimated_graph)
            full_augmenter.prepare(data_ref, data_aug, self.method_config.weight_kernel_cfg)
            weights = full_augmenter.augment()
            self.augmenter = full_augmenter
        else:
            raise NotImplementedError()
        return weights


if __name__ == '__main__':
    import doctest
    doctest.testmod()
