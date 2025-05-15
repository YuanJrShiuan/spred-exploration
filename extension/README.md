Here we replace Hadamard reparametrization with alternatives:
1. Additive Product
   As an alternative to the Hadamard product used in the original \textit{spred} formulation, we explore an additive structure defined as:
\[V = U + W\].
Based on our theoretical analysis and experiment, the zero rate remains flat at 0, which indicates that sparsity is not induced under any setting. These results suggest that additive composition disrupts the sparsity-inducing effect inherent in the original reparameterization. Despite its simplicity, the additive structure is not viable for structured sparse learning.
2. sign-root product
   Defined as:
\[
V_i = \|U \odot W\|_1^{1/2} \cdot \mathrm{sgn}(U_i) \cdot |U_i|^{1/2} \cdot \mathrm{sgn}(W_i) \cdot |W_i|^{1/2}
\]
We denote this transformation by $V = U \circledcirc W$.
This variant aims to preserve theoretical alignment between the relaxed and target objectives. Substituting $U \circledcirc W$ into the surrogate and $\ell_1$ formulations yields:
\begin{align*}
L_{rs}(W, U) &= L(U \circledcirc W) + 2\kappa \|U \odot W\|_1 \\
L_{L1}(V)&=L(U\circledcirc W)+2\kappa\|U \circledcirc W\|_{1}\\
		&=L(U\circledcirc W)+2\kappa\|U \odot W\|_{1} \times \frac{\Sigma_{i}|U_{i}W_{i}|^{1/2}}{\|U\odot W\|_{1}^{1/2}}
\end{align*}
where $\lambda(U, W) = \frac{\Sigma_{i}|U_{i}W_{i}|^{1/2}}{\|U\odot W\|_{1}^{1/2}} \geq 1$ by the Cauchy-Schwarz inequality. Thus, $L_{rs}(W,U) \leq L_{L1}(V)$, minimizing $L_{rs}(W,U)$ approximates minimizing a lower bound of $L_{L_1}(V)$, which may lead to suboptimal sparsification or even fail to induce sparsity. 
By our experiments, while the $\ell_1$ loss decreases steadily with increasing $\alpha$ (i.e., the \(\kappa\) in formula), the zero rate remains negligible. This suggests that the decay process is overly smoothed, and sparsification fails to emerge.
3. SVD product
