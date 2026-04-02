**Convergence Guarantee for the Informed Prior**

**Assumptions.** (1) Score estimator one-sided Lipschitz: $\langle s\_\theta(\mathbf{y}\_1, \hat{\mathbf{Y}}, t) - s\_\theta(\mathbf{y}\_2, \hat{\mathbf{Y}}, t), \mathbf{y}\_1 - \mathbf{y}\_2\rangle \leq l\_2(t)\lVert\mathbf{y}\_1 - \mathbf{y}\_2\rVert\_2^2$. (2) Regularity: $q\_t, p\_t$ are positive, $C^2$, rapidly decaying densities.

**Step 1. SDE Formulation.** ZeroDiff's discrete forward process $\mathbf{Y}\_t = \sqrt{\alpha\_t} \mathbf{Y}\_{t-1} + (1 - \sqrt{\alpha\_t})\hat{\mathbf{Y}} + \sqrt{\beta\_t}\boldsymbol{\epsilon}\_t$ corresponds in continuous time to:

Forward: $d\mathbf{y} = -\frac{\beta\_t}{2}(\mathbf{y} - \hat{\mathbf{Y}})dt + \sqrt{\beta\_t} dw$

Reverse (true score): $d\mathbf{y} = -[\frac{\beta\_t}{2}(\mathbf{y} - \hat{\mathbf{Y}}) - \beta\_t \nabla\_\mathbf{y} \log q(\mathbf{y}\_t \mid \hat{\mathbf{Y}})]dt + \sqrt{\beta\_t} d\bar{w}$

Reverse (learned): $d\mathbf{y} = -[\frac{\beta\_t}{2}(\mathbf{y} - \hat{\mathbf{Y}}) - \beta\_t s\_\theta(\mathbf{y}\_t, \hat{\mathbf{Y}}, t)]dt + \sqrt{\beta\_t} d\bar{w}$

Convention: forward $dt$ from $0$ to $T$; reverse from $T$ to $0$, both positive (CARD convention).


**Step 2. Continuity Equations.** Via standard Fokker-Planck analysis of the SDEs in Step 1, with $q\_t := q(\mathbf{y}\_t \mid \hat{\mathbf{Y}})$, $p\_t := p(\mathbf{y}\_t \mid \hat{\mathbf{Y}})$, the velocity fields are:

$\mathbf{v}\lbrack q\_t\rbrack(\mathbf{y}) = -\frac{\beta\_t}{2}(\mathbf{y} - \hat{\mathbf{Y}}) - \frac{\beta\_t}{2}\nabla \log q\_t(\mathbf{y})$

$\mathbf{v}\lbrack p\_t\rbrack(\mathbf{z}) = (-\frac{1}{2}(\mathbf{z} - \hat{\mathbf{Y}}) + s\_\theta(\mathbf{z}, \hat{\mathbf{Y}}, t))\beta\_t + \frac{\beta\_t}{2}\nabla \log p\_t(\mathbf{z})$

satisfying $\partial\_t q\_t + \nabla \cdot (q\_t \mathbf{v}\lbrack q\_t\rbrack) = 0$ and $\partial\_t p\_t + \nabla \cdot (p\_t \mathbf{v}\lbrack p\_t\rbrack) = 0$.

**Step 3. $W\_2$ Time Derivative.** Let $\pi\_t$ be the optimal coupling between $q\_t$ and $p\_t$. Under Assumption 3, the SDE solutions have smooth densities with finite second moments, forming absolutely continuous curves in $\mathcal{P}\_2$. By optimal transport theory (Ambrosio et al., 2005):

$\frac{1}{2}\frac{d}{dt} W\_2^2(q\_t, p\_t) = \mathbb{E}\_{\pi\_t}[(\mathbf{y} - \mathbf{z}) \cdot (\mathbf{v}\lbrack q\_t\rbrack(\mathbf{y}) - \mathbf{v}\lbrack p\_t\rbrack(\mathbf{z}))]$

Since $\frac{1}{2}\frac{d}{dt} W\_2^2 = W\_2 \frac{d}{dt} W\_2$, we get $-W\_2 \frac{d}{dt} W\_2 = -\mathbb{E}\_{\pi\_t}[(\mathbf{y} - \mathbf{z}) \cdot (\mathbf{v}\lbrack q\_t\rbrack - \mathbf{v}\lbrack p\_t\rbrack)]$.

**Step 4. Term-by-Term Estimation.** Expanding $\mathbf{v}\lbrack q\_t\rbrack - \mathbf{v}\lbrack p\_t\rbrack$ (note $\hat{\mathbf{Y}}$ cancels), and decomposing:

$-\mathbb{E}\_{\pi\_t}[(\mathbf{y} - \mathbf{z}) \cdot (\mathbf{v}\lbrack q\_t\rbrack - \mathbf{v}\lbrack p\_t\rbrack)] = D\_1 + D\_2 + D\_3$

where $D\_1 = \mathbb{E}\_{\pi\_t}[\frac{\beta\_t}{2}\lVert\mathbf{y} - \mathbf{z}\rVert^2]$, $D\_2 = \beta\_t \mathbb{E}\_{\pi\_t}[(\mathbf{y} - \mathbf{z}) \cdot (s\_\theta(\mathbf{z}) - \nabla \log q\_t(\mathbf{y}))]$, $D\_3 = \frac{\beta\_t}{2} \mathbb{E}\_{\pi\_t}[(\mathbf{y} - \mathbf{z}) \cdot (\nabla \log p\_t(\mathbf{z}) - \nabla \log q\_t(\mathbf{y}))]$.

$D\_1 = \frac{\beta\_t}{2} W\_2^2$ since $\frac{\beta\_t}{2}$ is scalar. For $D\_2$, adding/subtracting $s\_\theta(\mathbf{y})$ and applying Assumption 2 + Cauchy-Schwarz: $D\_2 \leq \beta\_t l\_2(t) W\_2^2 + \beta\_t W\_2 \sqrt{H(t)}$, where $H(t) = \mathbb{E}\_{q\_t}[\lVert s\_\theta - \nabla \log q\_t\rVert^2]$. By Kwon et al. (2022) Lemma 2, $D\_3 \leq 0$. Discarding $D\_3$ and dividing by $W\_2 > 0$:

$-\frac{d}{dt} W\_2 \leq (\frac{\beta\_t}{2} + \beta\_t l\_2(t)) W\_2 + \beta\_t \sqrt{H(t)} \quad (I)$

**Step 5. Gronwall's Inequality.** Define $M(t) = \exp\lbrace\int\_0^t (\frac{\beta\_s}{2} + l\_2(s)\beta\_s) ds\rbrace$, so $\frac{d}{dt} M = (l\_1 + \beta\_t l\_2) M$ and $M(0) = 1$. Multiplying $(I)$ by $M(t)$:

$-M(t)\frac{d}{dt} W\_2 \leq (l\_1 + \beta\_t l\_2) M W\_2 + \beta\_t M \sqrt{H(t)}$

Since $-\frac{d}{dt}[M W\_2] = -M\frac{d}{dt} W\_2 - (l\_1 + \beta\_t l\_2) M W\_2$, substituting and canceling:

$-\frac{d}{dt}[M(t) W\_2(q\_t, p\_t)] \leq \beta\_t M(t) \sqrt{H(t)}$

Integrating from $0$ to $T$ with $M(0) = 1$:

$W\_2(q\_0, p\_0) \leq \int\_0^T \beta\_t M(t) \sqrt{H(t)} dt + M(T) W\_2(q\_T, p\_T) \quad (II)$

The first term depends only on score estimation quality and is prior-independent. The second term is the initialization error controlled by the informed prior. We now bound it.

**Step 6. Initialization Error.** Here $q\_T = q(\mathbf{Y}\_T \mid \hat{\mathbf{Y}})$ is a Gaussian mixture (marginal over $\mathbf{Y}\_0$), not a single Gaussian. We construct an explicit coupling for an upper bound. Sample $\mathbf{Y}\_0 \sim q(\mathbf{Y}\_0)$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, let:

$\mathbf{Y}\_T = \sqrt{\bar\alpha\_T}\mathbf{Y}\_0 + (1 - \sqrt{\bar\alpha\_T})\hat{\mathbf{Y}} + \sqrt{\bar\sigma\_T}\boldsymbol{\epsilon} \sim q\_T$, $\quad \mathbf{Z}\_T = \hat{\mathbf{Y}} + \sqrt{\bar\sigma\_T}\boldsymbol{\epsilon} \sim p\_T$

Same noise $\boldsymbol{\epsilon}$ cancels: $\lVert\mathbf{Y}\_T - \mathbf{Z}\_T\rVert^2 = \bar\alpha\_T \lVert\mathbf{Y}\_0 - \hat{\mathbf{Y}}\rVert^2$. Any coupling upper-bounds $W\_2$:

$W\_2^2(q\_T, p\_T) \leq \bar\alpha\_T \cdot \mathbb{E}[\lVert\mathbf{Y}\_0 - \hat{\mathbf{Y}}\rVert^2]$

Substituting into $(II)$:

$W\_2(q\_0, p\_0) \leq \int\_0^T \beta\_t M(t) \sqrt{H(t)} dt + M(T)\sqrt{\bar\alpha\_T} \cdot \sqrt{\mathbb{E}[\lVert\mathbf{Y}\_0 - \hat{\mathbf{Y}}\rVert^2]}$

Standard DDPM ($\hat{\mathbf{Y}} = \mathbf{0}$) gives second term $M(T)\sqrt{\bar\alpha\_T} \cdot \sqrt{\mathbb{E}[\lVert\mathbf{Y}\_0\rVert^2]}$. Any prior with $\mathbb{E}[\lVert\mathbf{Y}\_0 - \hat{\mathbf{Y}}\rVert^2] < \mathbb{E}[\lVert\mathbf{Y}\_0\rVert^2]$ strictly tightens the bound. In ZeroDiff, $\hat{\mathbf{Y}} = \hat{\mu} + \hat{\sigma} \cdot f\_\omega(\mathbf{X})$ from VAE moment estimation + dynamics learning satisfies this condition, providing a strictly tighter guarantee. 

**Remark.** The first term (accumulated score error) vanishes as $H(t) \to 0$ with sufficient denoiser training, independent of prior choice. The second term (initialization error) is directly reduced by the informed prior through $\mathbb{E}[\lVert\mathbf{Y}\_0 - \hat{\mathbf{Y}}\rVert^2]$, which is the mechanism by which ZeroDiff's moment estimation and dynamics learning improve generation quality.
