from abc import abstractmethod
from typing import Dict
import math
import torch
from torch import nn
from torch.nn import init
from gram import gram_schmidt

class MyModelMixin:
    @abstractmethod
    def get_weights(self) -> Dict:
        pass

class Linear(nn.Linear, MyModelMixin):
    def get_weights(self):
        if self.bias:
            return {
                'weight': self.weight,
                'bias': self.bias
            }
        else:
            return {'weight': self.weight}

class SpaRedLinear(nn.Module, MyModelMixin):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpaRedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.weight2 = nn.Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(
                out_features, **factory_kwargs))
            self.bias2 = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.bias2, -bound, bound)

### Matrix Multiplication Product
#     def forward(self, X, **kwargs):
#         U1 = gram_schmidt(self.weight)
#         U2 = gram_schmidt(self.weight2)
#         weight = U1 @ U2.T
#         if self.use_bias:
#             bias = self.bias * self.bias2
#             x = nn.functional.linear(X, weight, bias)
#         else:
#             x = nn.functional.linear(X, weight)
#         assert not torch.isnan(x).any()
#         return x
#
#     def get_weights(self):
#         U1 = gram_schmidt(self.weight)
#         U2 = gram_schmidt(self.weight2)
#         if self.use_bias:
#             return {'weight': U1 @ U2.T,
#                     'bias': self.bias * self.bias2}
#         else:
#             return {'weight': U1 @ V2.T}


###Sign-Cubic-root Product
#     def forward(self, X, **kwargs):
#         weight_sign=torch.sign(self.weight)
#         weight2_sign = torch.sign(self.weight2)
#         abs= torch.abs(self.weight)
#         abs2 = torch.abs(self.weight2)
#         weight = weight_sign * weight2_sign * abs.pow(1/3) * abs2.pow(1/3)
#         if self.use_bias:
#             bias_sign = torch.sign(self.bias)
#             bias2_sign = torch.sign(self.bias2)
#             bias_abs = torch.abs(self.bias)
#             bias_abs2 = torch.abs(self.bias2)
#             bias = bias_sign * bias2_sign * bias_abs.pow(1/3) * bias_abs2.pow(1/3)
#             x = nn.functional.linear(X, weight, bias)
#         else:
#             x = nn.functional.linear(X, weight)
#         assert not torch.isnan(x).any()
#         return x
#
#     def get_weights(self):
#         weight_sign = torch.sign(self.weight)
#         weight2_sign = torch.sign(self.weight2)
#         abs = torch.abs(self.weight)
#         abs2 = torch.abs(self.weight2)
#         if self.use_bias:
#             bias_sign = torch.sign(self.bias)
#             bias2_sign = torch.sign(self.bias2)
#             bias_abs = torch.abs(self.bias)
#             bias_abs2 = torch.abs(self.bias2)
#             return {'weight': weight_sign * weight2_sign * abs.pow(1/3) * abs2.pow(1/3),
#                     'bias': bias_sign * bias2_sign * bias_abs.pow(1/3) * bias_abs2.pow(1/3)}
#         else:
#             return {'weight': weight_sign * weight2_sign * abs.pow(1/3) * abs2.pow(1/3)}




####################dense+sparse
# class SpaRedLinear(nn.Module, MyModelMixin):
#     def __init__(self, input_dim, output_dim, bias=True, device=None, dtype=None):
#         """
#         参数说明：
#         - input_dim: 输入特征数
#         - output_dim: 输出特征数
#         - bias: 是否使用偏置
#         - 本模型采用分离设计：
#           * dense 部分（weight_dense、bias_dense）用于捕捉主要信号
#           * sparse 部分（weight_sparse、bias_sparse）将在训练中通过 L1 正则化鼓励稀疏
#         """
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(SpaRedLinear, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.use_bias = bias
#
#         # dense 部分：主要信号
#         self.weight_dense = nn.Parameter(torch.empty((output_dim, input_dim), **factory_kwargs))
#         # sparse 部分：用于施加稀疏性正则，初始值设置为 1 保证初始时不改变 dense 部分效果
#         self.weight_sparse = nn.Parameter(torch.empty((output_dim, input_dim), **factory_kwargs))
#
#         if bias:
#             self.bias_dense = nn.Parameter(torch.empty(output_dim, **factory_kwargs))
#             self.bias_sparse = nn.Parameter(torch.empty(output_dim, **factory_kwargs))
#         else:
#             self.register_parameter('bias_dense', None)
#             self.register_parameter('bias_sparse', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         # 对 dense 部分使用 kaiming_uniform 初始化
#         init.kaiming_uniform_(self.weight_dense, a=math.sqrt(5))
#         # 将 sparse 部分初始化为 1.0，初始时有效权重=weight_dense*1=weight_dense
#         init.constant_(self.weight_sparse, 1.0)
#
#         if self.use_bias:
#             bound = 1 / math.sqrt(self.input_dim)
#             init.uniform_(self.bias_dense, -bound, bound)
#             init.constant_(self.bias_sparse, 1.0)
#
#     def forward(self, x):
#         # 有效权重为 dense 和 sparse 部分的乘积
#         effective_weight = self.weight_dense * self.weight_sparse
#         if self.use_bias:
#             effective_bias = self.bias_dense * self.bias_sparse
#             return nn.functional.linear(x, effective_weight, effective_bias)
#         else:
#             return nn.functional.linear(x, effective_weight)
#
#     def get_weights(self):
#         """
#         返回当前计算出的有效权重和偏置，训练函数中会调用该函数计算 L1 正则项和评估指标
#         """
#         effective_weight = self.weight_dense * self.weight_sparse
#         if self.use_bias:
#             effective_bias = self.bias_dense * self.bias_sparse
#             return {'weight': effective_weight, 'bias': effective_bias}
#         else:
#             return {'weight': effective_weight}




###Additive Product
#     def forward(self, X, **kwargs):
#         weight = self.weight + self.weight2
#         if self.use_bias:
#             bias = self.bias + self.bias2
#             x = nn.functional.linear(X, weight, bias)
#         else:
#             x = nn.functional.linear(X, weight)
#         assert not torch.isnan(x).any()
#         return x
#
#     def get_weights(self):
#         if self.use_bias:
#             return {'weight': self.weight + self.weight2,
#                     'bias': self.bias + self.bias2}
#         else:
#             return {'weight': self.weight + self.weight2}

###Sign-Root Product
    def _compute_weights(self):
        # """计算组合权重矩阵"""
        # 特征向量点积 U.W (形状 [out_features])
        UW = self.weight * self.weight2
        dot_products=torch.norm(UW,p=1)
        # # 计算 |U.W|^(2/3) (形状 [out_features, 1])
        term1 =(dot_products).pow(1 / 2)

        # 计算 U^(1/2) ∘ W^(1/2) (保留符号)
        U_sign = torch.sign(self.weight)
        W_sign = torch.sign(self.weight2)
        U_mag = torch.abs(self.weight)
        W_mag = torch.abs(self.weight2)
        term2 = U_sign * U_mag.pow(1 / 2) * W_sign * W_mag.pow(1 / 2)

        return term1* term2  # 广播乘法 [out_features,1] * [out_features,in_features]

    def _compute_bias(self):
        """计算组合偏置 (同权重逻辑)"""
        dot_bias = self.bias * self.bias2
        term1_bias = torch.abs(dot_bias ).pow(1 / 2)

        b_sign1 = torch.sign(self.bias)
        b_sign2 = torch.sign(self.bias2)
        b_mag1 = torch.abs(self.bias)
        b_mag2 = torch.abs(self.bias2)
        term2_bias = b_sign1 * b_mag1.pow(1 / 2) * b_sign2 * b_mag2.pow(1 / 2)

        return  term2_bias

    def forward(self, X):
        combined_weight = self._compute_weights()
        combined_bias = self._compute_bias() if self.use_bias else None
        return nn.functional.linear(X, combined_weight, combined_bias)

    def get_weights(self):
        combined_weight = self._compute_weights()
        if self.use_bias:
            return {'weight': combined_weight, 'bias': self._compute_bias()}
        else:
            return {'weight': combined_weight}

class SparedLinearRegression(nn.Module, MyModelMixin):
    def __init__(
            self,
            input_dim,
            output_dim,
            bias=False):
        super(SparedLinearRegression, self).__init__()
        torch.manual_seed(111)
        self.linear = SpaRedLinear(input_dim, output_dim, bias)

    def forward(self, x):
        return self.linear(x)

    def get_weights(self):
        return self.linear.get_weights()
