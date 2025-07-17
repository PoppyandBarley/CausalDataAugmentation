import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union
from tqdm import tqdm
from statsmodels.nonparametric import kernels
from statsmodels.sandbox.nonparametric import kernels as sandbox_kernels
from statsmodels.nonparametric import bandwidths
from causalaug.code.base import WeightComputer
# from base import WeightComputer


# 假设的类型定义
CONTINUOUS = 'c'
DISCRETE = 'u'
MINIMUM_CONTI_BANDWIDTH = 1e-100

import numpy as np
from typing import Callable, List, Tuple

# 支持的核函数
def gaussian_kernel(bw, x, y):
    return np.exp(-0.5 * ((x - y) / bw) ** 2)

def indicator_kernel(_, x, y):
    return (x == y).astype(float)

kernel_func = dict(
    wangryzin=kernels.wang_ryzin,
    aitchisonaitken=kernels.aitchison_aitken,

    # https://tedboy.github.io/statsmodels_doc/_modules/statsmodels/nonparametric/kernels.html#gaussian
    gaussian=kernels.gaussian,
    aitchison_aitken_reg=kernels.aitchison_aitken_reg,
    wangryzin_reg=kernels.wang_ryzin_reg,
    gauss_convolution=kernels.gaussian_convolution,
    wangryzin_convolution=kernels.wang_ryzin_convolution,
    aitchisonaitken_convolution=kernels.aitchison_aitken_convolution,
    gaussian_cdf=kernels.gaussian_cdf,
    aitchisonaitken_cdf=kernels.aitchison_aitken_cdf,
    wangryzin_cdf=kernels.wang_ryzin_cdf,
    d_gaussian=kernels.d_gaussian,
    # Following are added here:
    indicator=indicator_kernel
)

# 类型标识
CONTINUOUS = 'c'
DISCRETE = 'd'

def bw_normal_reference(x: np.ndarray, kernel=sandbox_kernels.Gaussian) -> float:
    """
    Plug-in bandwidth with kernel specific constant based on normal reference.
    This bandwidth minimizes the mean integrated square error if the true
    distribution is the normal. This choice is an appropriate bandwidth for
    single peaked distributions that are similar to the normal distribution.

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Used to calculate the constant for the plug-in bandwidth.

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns C * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))
       C = constant from Hansen (2009)

    When using a Gaussian kernel this is equivalent to the 'scott' bandwidth up
    to two decimal places. This is the accuracy to which the 'scott' constant is
    specified.

    References
    ----------

    Silverman, B.W. (1986) `Density Estimation.`
    Hansen, B.E. (2009) `Lecture Notes on Nonparametrics.`
    """
    C = kernel().normal_reference_constant
    A = bandwidths._select_sigma(x)
    n = len(x)
    return C * A * n**(-0.2)


class BandwidthNormalReference:
    """Class to propose the rule-of-thumb bandwidth."""
    def __init__(self, coeff:float=1):
        """Constructor.

        Parameters:
            coeff : Coefficient to multiply the rule-of-thumb bandwidth.
        """
        self.coeff = coeff

    def __call__(self, *args, **kwargs) -> float:
        """Compute the bandwidth.

        Returns:
            Computed bandwidth.
        """
        return self.coeff * bw_normal_reference(*args, **kwargs)

class VanillaProductKernelConfig:
    """A configuration set used for product kernels.

    Parameters:
        conti_kertype : Default: 'gaussian'.
        dis_kertype : statmodels' original default is 'wangryzin'.
        conti_bw_method :
        dis_bw_method :
    """
    conti_kertype: str = 'gaussian'
    conti_bw_method: str = 'normal_reference'
    conti_bw_temperature: float = 1.

    dis_kertype: str = 'indicator'
    dis_bw_method: str = 'indicator'

class VanillaProductKernel:

    BW_METHODS = {
        'normal_reference': BandwidthNormalReference(),
        'indicator': lambda x: None,
    }

    def __init__(self, z_ref: np.ndarray, ztype: str, mp_ref: np.ndarray, vartypes: str, config: VanillaProductKernelConfig):
        self.z_ref = z_ref
        self.ztype = ztype
        self.mp_ref = mp_ref  # shape: (n_samples, n_dims)
        self.vartypes = vartypes
        self.config = config
        self.kernels = dict(c=config.conti_kertype,
                            u=config.dis_kertype)
        self.bw_methods = dict(c=config.conti_bw_method,
                               u=config.dis_bw_method)
        
        self.kernel_bw: List[Callable, np.ndarray] = []


        self._build_kernels()

    def _build_kernels(self):
        bw_method_z = 'indicator'
        # bw_method_z = self.bw_methods.get(self.ztype, lambda x: 'not implemented')
        bw_z = self.BW_METHODS[bw_method_z](self.z_ref)
        # if self.ztype == 'c':
        #     # 计算带宽
        #     bw_z = bw_z * self.config.conti_bw_temperature
        #     if bw_z == 0:
        #         # Error handling for the case that there is only one unique value for the variable in the data.
        #         bw_z = MINIMUM_CONTI_BANDWIDTH

        #     elif self.ztype == DISCRETE:
        #         bw_z = None
        # kernel_z = kernel_func[self.kernels[self.ztype]]
        kernel_z = kernel_func['indicator']
        self.kernel_bw_z = (kernel_z, bw_z)

        for k, vtype in enumerate(self.vartypes):
            col_data = self.mp_ref[:, k]
            bw_method = self.bw_methods.get(vtype, lambda x: 'not implemented')
            bw = self.BW_METHODS[bw_method](col_data)
            if vtype == 'c':
                # 计算带宽
                bw = bw * self.config.conti_bw_temperature
                if bw == 0:
                    # Error handling for the case that there is only one unique value for the variable in the data.
                    bw = MINIMUM_CONTI_BANDWIDTH

            elif vtype == DISCRETE:
                bw = None
            else:
                raise ValueError(f"Unsupported variable type '{vtype}' at index {k}")
            
            kernel_ = kernel_func[self.kernels[vtype]]
            self.kernel_bw.append((kernel_, bw))
    
    def __call__(self, z_new: np.ndarray, data_new: np.ndarray) -> np.ndarray:
        """
        Parameters:
            z_new : shape (n_batch, 1)
            data_new : shape (n_batch, n_dims) — 新数据
            normalize : bool — 是否对结果按行归一化为概率

        Returns:
            weights : shape (n_batch, n_ref_samples)
        """

        kernel_z, bw_z = self.kernel_bw_z
        y_val = kernel_z(bw_z, z_new[:, None], self.z_ref[None, :])
        weights_list = []

        for k, _ in enumerate(self.vartypes):
            kernel_f, bw = self.kernel_bw[k]
            x_new = data_new[:, k][:, None]           # (n_batch, 1)
            x_ref = self.mp_ref[:, k][None, :]  # (1, n_ref)

            if kernel_f:
                k_vals = kernel_f(bw, x_new, x_ref)   # (n_batch, n_ref)
            else:
                raise ValueError("Unsupported kernel function")

            weights_list.append(k_vals)

        # 元素逐项相乘（对应 product kernel）
        combined_denom = np.prod(weights_list, axis=0)  # shape: (n_batch, n_ref)
        combined_num = y_val*combined_denom

        return combined_denom, combined_num

class KernelWeightComputer(WeightComputer):
    """
    计算新数据与参考数据之间，在某一列上的核相似度（权重）
    """

    def __init__(self, kernel: VanillaProductKernel):
        self.kernel = kernel

    def __call__(self, z_new, data_new):
        _gram_matrix, _num_matrix = self.kernel(z_new=z_new, data_new=data_new)
        divisor = _gram_matrix.sum(axis=1, keepdims=True)
        divisor_safe = np.where(divisor == 0, 1e-12, divisor)
        # Without "out" argument, where the "where" condition is not met, the value is not initialized.
        weights = np.divide(_num_matrix,
                            divisor_safe,
                            where=divisor_safe != 0,
                            out=np.zeros(_gram_matrix.shape))
        # # with np.errstate(divide='ignore'):
        # assert np.sum(np.isnan(weights)) == 0

        # # All rows should have values that sum to either 1 or 0
        # rowsums = weights.sum(axis=1)
        # zero_or_one = np.logical_or(np.isclose(rowsums, 1.),
        #                             np.isclose(rowsums, 0.))
        # assert np.sum(np.logical_not(zero_or_one)) == 0
        return weights.sum(axis=1).reshape(-1,1)  # shape(n_new, n_ref)

    




class UnconditionalWeightComputer:
    def __init__(self, n):
        self.n = n
    
    def __call__(self, _=None, d=None) -> np.ndarray:
        """Return the uniform weight.

        Returns:
            The uniform weight array (each being equal to 1/n) for the number of the candidates.
        """
        # return np.ones((1, self.n))
        return np.ones((self.n, 1)) / self.n
    

if __name__ == '__main__':
    config = VanillaProductKernelConfig()
    data = np.random.rand(2, 3)
    z = np.random.rand(2, )
    data_new = np.random.rand(2, 3)
    z_new = np.random.rand(2, )
    _kernel = VanillaProductKernel(z, 'c', data, 'cc', config)
    _kernel_weight_computer = KernelWeightComputer(_kernel)
    weights = _kernel_weight_computer(z_new, data_new)
    print(weights)