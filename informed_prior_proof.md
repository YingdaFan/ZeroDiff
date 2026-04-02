**Convergence Guarantee for the Informed Prior**

**Assumptions.** (1) Score estimator one-sided Lipschitz: $\langle s_\theta(\mathbf{y}_1, \hat{\mathbf{Y}}, t) - s_\theta(\mathbf{y}_2, \hat{\mathbf{Y}}, t), \mathbf{y}_1 - \mathbf{y}_2\rangle \leq l_2(t)\lVert\mathbf{y}_1 - \mathbf{y}_2\rVert_2^2$. (2) Regularity: $q_t, p_t$ are positive, $C^2$, rapidly decaying densities with finite second moments for all $t \in [0,T]$. The velocity fields $\mathbf{v}[q_t], \mathbf{v}[p_t]$ are $L^2$-integrable with respect to their respective densities. Under these conditions, $t \mapsto q_t$ and $t \mapsto p_t$ are absolutely continuous curves in $(\mathcal{P}_2(\mathbb{R}^d), W_2)$.

**Step 1. SDE Formulation.** ZeroDiff's discrete forward process $\mathbf{Y}_t = \sqrt{\alpha_t} \mathbf{Y}_{t-1} + (1 - \sqrt{\alpha_t})\hat{\mathbf{Y}} + \sqrt{\beta_t}\boldsymbol{\epsilon}_t$ corresponds in continuous time to:

Forward: $d\mathbf{y} = -\frac{\beta_t}{2}(\mathbf{y} - \hat{\mathbf{Y}})dt + \sqrt{\beta_t}\, dw$

Reverse (true score): $d\mathbf{y} = -\left[\frac{\beta_t}{2}(\mathbf{y} - \hat{\mathbf{Y}}) - \beta_t \nabla_\mathbf{y} \log q(\mathbf{y}_t \mid \hat{\mathbf{Y}})\right]dt + \sqrt{\beta_t}\, d\bar{w}$

Reverse (learned): $d\mathbf{y} = -\left[\frac{\beta_t}{2}(\mathbf{y} - \hat{\mathbf{Y}}) - \beta_t s_\theta(\mathbf{y}_t, \hat{\mathbf{Y}}, t)\right]dt + \sqrt{\beta_t}\, d\bar{w}$

Convention: forward $dt$ from $0$ to $T$; reverse from $T$ to $0$, both positive (CARD convention).


**Step 2. Continuity Equations.** Via standard Fokker-Planck analysis of the SDEs in Step 1, with $q_t := q(\mathbf{y}_t \mid \hat{\mathbf{Y}})$, $p_t := p(\mathbf{y}_t \mid \hat{\mathbf{Y}})$. Throughout, $t \in [0, T]$ indexes the noise level: $t = 0$ corresponds to data and $t = T$ to the noise limit. Both $q_t$ (forward) and $p_t$ (learned reverse) are defined as densities at noise level $t$, with the reverse SDE re-parameterized as positive $dt$ from $T$ to $0$ following the CARD convention. The velocity fields are:

$$\mathbf{v}[q_t](\mathbf{y}) = -\frac{\beta_t}{2}(\mathbf{y} - \hat{\mathbf{Y}}) - \frac{\beta_t}{2}\nabla \log q_t(\mathbf{y})$$

$$\mathbf{v}[p_t](\mathbf{z}) = \left(-\frac{1}{2}(\mathbf{z} - \hat{\mathbf{Y}}) + s_\theta(\mathbf{z}, \hat{\mathbf{Y}}, t)\right)\beta_t + \frac{\beta_t}{2}\nabla \log p_t(\mathbf{z})$$

satisfying $\partial_t q_t + \nabla \cdot (q_t \mathbf{v}[q_t]) = 0$ and $\partial_t p_t + \nabla \cdot (p_t \mathbf{v}[p_t]) = 0$.

**Step 3. $W_2$ Time Derivative.** Let $\pi_t$ be the optimal coupling between $q_t$ and $p_t$. By Assumption 2, both $t \mapsto q_t$ and $t \mapsto p_t$ are absolutely continuous curves in $(\mathcal{P}_2(\mathbb{R}^d), W_2)$ with $L^2$-integrable velocity fields. The Benamou–Brenier differentiation formula (Ambrosio et al., 2005, Theorem 8.4.7) then gives:

$$\frac{1}{2}\frac{d}{dt} W_2^2(q_t, p_t) = \mathbb{E}_{\pi_t}\left[(\mathbf{y} - \mathbf{z}) \cdot (\mathbf{v}[q_t](\mathbf{y}) - \mathbf{v}[p_t](\mathbf{z}))\right]$$

Since $\frac{1}{2}\frac{d}{dt} W_2^2 = W_2 \frac{d}{dt} W_2$, we get $-W_2 \frac{d}{dt} W_2 = -\mathbb{E}_{\pi_t}\left[(\mathbf{y} - \mathbf{z}) \cdot (\mathbf{v}[q_t] - \mathbf{v}[p_t])\right]$.

**Step 4. Term-by-Term Estimation.** Expanding $\mathbf{v}[q_t] - \mathbf{v}[p_t]$ (note $\hat{\mathbf{Y}}$ cancels), and decomposing:

$$-\mathbb{E}_{\pi_t}\left[(\mathbf{y} - \mathbf{z}) \cdot (\mathbf{v}[q_t] - \mathbf{v}[p_t])\right] = D_1 + D_2 + D_3$$

where $D_1 = \mathbb{E}_{\pi_t}\left[\frac{\beta_t}{2}\lVert\mathbf{y} - \mathbf{z}\rVert^2\right]$, $D_2 = \beta_t \mathbb{E}_{\pi_t}\left[(\mathbf{y} - \mathbf{z}) \cdot (s_\theta(\mathbf{z}) - \nabla \log q_t(\mathbf{y}))\right]$, $D_3 = \frac{\beta_t}{2} \mathbb{E}_{\pi_t}\left[(\mathbf{y} - \mathbf{z}) \cdot (\nabla \log p_t(\mathbf{z}) - \nabla \log q_t(\mathbf{y}))\right]$.

$D_1 = \frac{\beta_t}{2} W_2^2$ since $\frac{\beta_t}{2}$ is scalar. For $D_2$, adding/subtracting $s_\theta(\mathbf{y})$ and applying Assumption 1 + Cauchy-Schwarz: $D_2 \leq \beta_t l_2(t) W_2^2 + \beta_t W_2 \sqrt{H(t)}$, where $H(t) = \mathbb{E}_{q_t}\left[\lVert s_\theta - \nabla \log q_t\rVert^2\right]$. By Kwon et al. (2022) Lemma 2, $D_3 \leq 0$. Discarding $D_3$ and dividing by $W_2 > 0$:

$$-\frac{d}{dt} W_2 \leq \left(\frac{\beta_t}{2} + \beta_t l_2(t)\right) W_2 + \beta_t \sqrt{H(t)} \quad \text{(I)}$$

**Step 5. Gronwall's Inequality.** Define $M(t) = \exp\left\lbrace\int_0^t \left(\frac{\beta_s}{2} + l_2(s)\beta_s\right) ds\right\rbrace$, so $\frac{d}{dt} M = \left(\frac{\beta_t}{2} + \beta_t l_2\right) M$ and $M(0) = 1$. Multiplying $\text{(I)}$ by $M(t)$:

$$-M(t)\frac{d}{dt} W_2 \leq \left(\frac{\beta_t}{2} + \beta_t l_2\right) M W_2 + \beta_t M \sqrt{H(t)}$$

Since $-\frac{d}{dt}[M W_2] = -M\frac{d}{dt} W_2 - \left(\frac{\beta_t}{2} + \beta_t l_2\right) M W_2$, substituting and canceling:

$$-\frac{d}{dt}\left[M(t) W_2(q_t, p_t)\right] \leq \beta_t M(t) \sqrt{H(t)}$$

Integrating from $0$ to $T$ with $M(0) = 1$:

$$W_2(q_0, p_0) \leq \int_0^T \beta_t M(t) \sqrt{H(t)}\, dt + M(T) W_2(q_T, p_T) \quad \text{(II)}$$

The first term depends only on score estimation quality and is prior-independent. The second term is the initialization error controlled by the informed prior. We now bound it.

**Step 6. Initialization Error.** Here $q_T = q(\mathbf{Y}_T \mid \hat{\mathbf{Y}})$ is a Gaussian mixture (marginal over $\mathbf{Y}_0$), not a single Gaussian. We construct an explicit coupling for an upper bound. Sample $\mathbf{Y}_0 \sim q(\mathbf{Y}_0)$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, let:

$$\mathbf{Y}_T = \sqrt{\bar\alpha_T}\mathbf{Y}_0 + (1 - \sqrt{\bar\alpha_T})\hat{\mathbf{Y}} + \sqrt{\bar\sigma_T}\boldsymbol{\epsilon} \sim q_T, \quad \mathbf{Z}_T = \hat{\mathbf{Y}} + \sqrt{\bar\sigma_T}\boldsymbol{\epsilon} \sim p_T$$

Same noise $\boldsymbol{\epsilon}$ cancels: $\lVert\mathbf{Y}_T - \mathbf{Z}_T\rVert^2 = \bar\alpha_T \lVert\mathbf{Y}_0 - \hat{\mathbf{Y}}\rVert^2$. Any coupling upper-bounds $W_2$:

$$W_2^2(q_T, p_T) \leq \bar\alpha_T \cdot \mathbb{E}\left[\lVert\mathbf{Y}_0 - \hat{\mathbf{Y}}\rVert^2\right]$$

Substituting into $\text{(II)}$:

$$W_2(q_0, p_0) \leq \int_0^T \beta_t M(t) \sqrt{H(t)}\, dt + M(T)\sqrt{\bar\alpha_T} \cdot \sqrt{\mathbb{E}\left[\lVert\mathbf{Y}_0 - \hat{\mathbf{Y}}\rVert^2\right]}$$

Standard DDPM ($\hat{\mathbf{Y}} = \mathbf{0}$) gives second term $M(T)\sqrt{\bar\alpha_T} \cdot \sqrt{\mathbb{E}\left[\lVert\mathbf{Y}_0\rVert^2\right]}$. Any prior with $\mathbb{E}\left[\lVert\mathbf{Y}_0 - \hat{\mathbf{Y}}\rVert^2\right] < \mathbb{E}\left[\lVert\mathbf{Y}_0\rVert^2\right]$ strictly tightens the bound. In ZeroDiff, $\hat{\mathbf{Y}} = \hat{\mu} + \hat{\sigma} \cdot f_\omega(\mathbf{X})$ from VAE moment estimation + dynamics learning satisfies this condition, providing a strictly tighter guarantee.

**Remark.** The first term (accumulated score error) vanishes as $H(t) \to 0$ with sufficient denoiser training, independent of prior choice. The second term (initialization error) is directly reduced by the informed prior through $\mathbb{E}\left[\lVert\mathbf{Y}_0 - \hat{\mathbf{Y}}\rVert^2\right]$, which is the mechanism by which ZeroDiff's moment estimation and dynamics learning improve generation quality.
