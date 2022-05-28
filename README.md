# VAE

## maths

### KL divergence between two multivariate Gaussians

需要用到二次型的数学期望：设随机变量 $\pmb y\in\mathbb R^D$, 数学期望 $\mathbb E[\pmb y] = \pmb \mu$, 协方差 $\text{cov} (\pmb y) = \mathbf \Sigma$，方阵 $\mathbf A\in\mathbb R^{D\times D}$ 为常量，则二次型 $\pmb y^{\rm T} \mathbf A \pmb y$ 的数学期望为

$$
\mathbb E [\pmb y^{\rm T} \mathbf A \pmb y]
= \text{tr}(\mathbf A \mathbf \Sigma)
+\pmb\mu \mathbf A \pmb\mu
$$

设真实分布
$$
p(\pmb z)
= \mathcal N(\pmb z | \pmb \mu_1, \mathbf \Sigma_1)
= \cfrac{1}{(2\pi)^{D/2}}
\cfrac{1}{|\mathbf \Sigma_1|^{1/2}}
\exp \left(
-\cfrac{1}{2}
(\pmb z - \pmb \mu_1)^{\rm T}
\mathbf \Sigma_1^{-1}
(\pmb z - \pmb \mu_1)
\right)
$$

变分分布
$$
q(\pmb z)
= \mathcal N(\pmb z | \pmb \mu_2, \mathbf \Sigma_2)
= \cfrac{1}{(2\pi)^{D/2}}
\cfrac{1}{|\mathbf \Sigma_2|^{1/2}}
\exp\left(
-\cfrac{1}{2}
(\pmb z - \pmb \mu_2)^{\rm T}
\mathbf \Sigma_2^{-1}
(\pmb z - \pmb \mu_2)
\right)
$$

KL散度
$$
\begin{aligned}
\text{KL}(q||p)
&= \mathbb E_{\pmb z\sim q(\pmb z)} \left[\ln \cfrac{q(\pmb z)}{p(\pmb z)}\right]\\
&= \cfrac{1}{2}
\ln \left(
\cfrac{|\mathbf \Sigma_1|}{|\mathbf \Sigma_2|}
\right)
+\cfrac{1}{2} \mathbb E \left[
(\pmb z - \pmb \mu_1)^{\rm T}
\mathbf \Sigma_1^{-1}
(\pmb z - \pmb \mu_1)
\right]\\
&\quad
-\cfrac{1}{2} \mathbb E \left[
(\pmb z - \pmb \mu_2)^{\rm T}
\mathbf \Sigma_1^{-2}
(\pmb z - \pmb \mu_2)
\right]\\
&= \cfrac{1}{2}
\ln \left(
\cfrac{|\mathbf \Sigma_1|}{|\mathbf \Sigma_2|}
\right)
-\cfrac{D}{2}
+\cfrac{1}{2}
\text{tr} (\mathbf \Sigma_1^{-1} \mathbf \Sigma_2)
+\cfrac{1}{2}
(\pmb \mu_2 - \pmb \mu_1)^{\rm T}
\mathbf \Sigma_1^{-1}
(\pmb \mu_2 - \pmb \mu_1)
\end{aligned}
$$

其中 $\pmb\mu_1 = \pmb 0, \mathbf\Sigma_1 = \mathbf I$，代入得

$$
\text{KL}(q||p)
= -\cfrac{1}{2}
\ln \left(
|\mathbf \Sigma_2|
\right) -
\cfrac{D}{2} +
\cfrac{1}{2}
\text{tr} (\mathbf \Sigma_2) +
\cfrac{1}{2}
\pmb \mu_2^{\rm T}
\pmb \mu_2
$$

对于一个minibatch，观测数据 $\mathbf X = \{\pmb x_1, \cdots, \pmb x_B\}$，经过编码器得到隐变量 $\mathbf Z = \{\pmb z_1, \cdots, \pmb z_B\}$，则可以用 $\mathbf Z$ 的样本均值和样本协方差来估计这里的 $\pmb\mu_2$ 和 $\mathbf\Sigma_2$。
