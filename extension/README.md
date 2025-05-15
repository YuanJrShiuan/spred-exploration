# Alternatives to Hadamard Reparametrization

In this work, we explore alternatives to the Hadamard product used in the original *spred* formulation.

---

## 1. Additive Product

As an alternative to the Hadamard product, we explore an additive structure:

$$
V = U + W
$$

Based on our theoretical analysis and experiments, the zero rate remains flat at 0, indicating that sparsity is not induced under any setting. These results suggest that additive composition disrupts the sparsity-inducing effect inherent in the original reparameterization. Despite its simplicity, the additive structure is **not viable** for structured sparse learning.

---

## 2. Sign-Root Product

We define the sign-root product as:

$$
V_i = \|U \odot W\|_1^{1/2} \cdot \mathrm{sgn}(U_i) \cdot |U_i|^{1/2} \cdot \mathrm{sgn}(W_i) \cdot |W_i|^{1/2}
$$

We denote this transformation as:

$$
V = U \circledcirc W
$$

This variant aims to preserve theoretical alignment between the relaxed and target objectives. Substituting $U \circledcirc W$ into the surrogate and $\ell_1$ formulations yields:

$$
L_{rs}(W, U) = L(U \circledcirc W) + 2\kappa \|U \odot W\|_1
$$

$$
L_{L1}(V) = L(U \circledcirc W) + 2\kappa \|U \circledcirc W\|_1 = L(U \circledcirc W) + 2\kappa \|U \odot W\|_1 \cdot \frac{\sum_i |U_i W_i|^{1/2}}{\|U \odot W\|_1^{1/2}}
$$

Let:

$$
\lambda(U, W) = \frac{\sum_i |U_i W_i|^{1/2}}{\|U \odot W\|_1^{1/2}} \geq 1 \quad \text{(by Cauchy-Schwarz inequality)}
$$

Then:

$$
L_{rs}(W, U) \leq L_{L1}(V)
$$

Minimizing $L_{rs}(W, U)$ approximates minimizing a lower bound of $L_{L1}(V)$, which may lead to suboptimal sparsification or even fail to induce sparsity.

**Experimental Observation:**  
While the $\ell_1$ loss decreases steadily with increasing $\alpha$ (i.e., the $\kappa$ in the formula), the zero rate remains negligible. This suggests that the decay process is overly smoothed, and sparsification fails to emerge.

---

## 3. SVD Product

(*To be detailed...*)

---

> ⚠️ **Note:** GitHub renders LaTeX only in `$...$` or `$$...$$` environments. For best visualization of derivations and math layout, consider viewing the full paper or [Jupyter notebook](./notebooks/reparam-alternatives.ipynb).

