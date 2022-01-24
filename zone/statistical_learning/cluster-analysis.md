[Hc's blog](https://hc1023.github.io/)

[Hc's blog: R与统计学习基础](https://hc1023.github.io/2021/06/13/Statistical-learning/)

[机器学习与人工智能 浙江大学 中国大学MOOC(慕课)](https://www.icourse163.org/course/ZJU-1206689820)

[TOC]

# 聚类分析

- 监督学习：有标签

- 无监督学习：给定数据，或者例子，但是没有标签

无监督学习可学习没有任何标签的聚类/群组，可应用于客户细分（即分组）、图像压缩、生物信息学。

另一类无监督学习：建模（模型的概率/参数）。例如学习翻译词典、语言的语法和社交图等。

聚类：通过对数据特征进行学习，将一组对象分组为类似对象的过程。

**聚类算法**

- 扁平算法：通常以一个随机的（部分的）划分开始
  - 迭代优化：K均值聚类、基于模型的聚类
  - 谱聚类
- 分层算法
  - 从上到下：凝聚
  - 从下到上：分裂

**硬聚类和软聚类**

硬聚类：每个数据只属于一个类

软聚类：一个数据可以属于多个类（概率），这对于创建可浏览 层次结构等应用程序更有意义

**谱聚类和K-均值算法**已经写过，可以见[Hc's blog: 聚类和主成分](https://hc1023.github.io/2019/12/13/cluster-PCA/)

这里再作稍许补充

- 参考高惠璇的《应用多元统计分析》，浙江大学mooc中所讲述的K-均值聚类其实为动态聚类中的**按批修改法**，按批修改法是当样本全部归类后才改变凝聚点。另一种自然的想法是对一个样品进行分类后，同时改变凝聚点，这就产生了**逐个修改法**，也被称作**K-均值法**。
- 距离的定义有很多，欧式距离适合空间数据，余弦距离适合文本数据。
- 谱聚类常使用**离差平方和法（WARD）**，谱聚类不要求数据高斯分布，没有**球形假设（K-均值）**或椭圆形假设（**EM聚类**）。

后文详细介绍EM聚类

## EM聚类：高斯混合模型（GMM）

假设数据从**高斯混合分布（椭圆数据）**产生，以一定的概率分配一个点到聚类中心，结果为分组的概率（**软聚类**）。

**EM算法**：存在“未观测”变量的情形下, 对模型参数进行估计。
未观测变量的学名是“隐变量”（latent variable）。令 $\mathbf{X}$ 表示已观测变量集，$\mathbf{Z}$ 表示隐变量集，$\Theta$ 表示模型参数。若欲对 $\Theta$ 做极大似然估计，则应最大化对数似然
$$
LL(\Theta \mid \mathbf{X}, \mathbf{Z})=\ln P(\mathbf{X}, \mathbf{Z} \mid \Theta).
$$
然而由于 $\mathbf{Z}$ 是隐变量，上式无法直接求解。此时我们可通过对 $\mathbf{Z}$ 计算期望，来最大化已观测数据的对数 “边际似然”(marginal likelihood)
$$
LL(\Theta \mid \mathbf{X})=\ln P(\mathbf{X} \mid \Theta)=\ln \sum_{\mathbf{Z}} P(\mathbf{X}, \mathbf{Z} \mid \Theta).
$$
EM (Expectation-Maximization) 算法 [Dempster et al., 1977$]$ 是常用的估计参数隐变量的利器，它是一种迭代式的方法，其基本想法是：

- 若参数 $\Theta$ 已知, 则可根据训练数据推断出最优隐变量 $\mathbf{Z}$ 的值 （$\mathrm{E}$ 步 ）；
- 反之 $,$ 若 $\mathbf{Z}$ 的值已知, 则可方便地对参数 $\Theta$ 做极大似然估计（$\mathrm{M}$ 步 ) 。

进一步，设共有$m$个样本
$$
\begin{aligned}
LL(\Theta \mid \mathbf{X}, \mathbf{Z})&=\ln P(\mathbf{X}, \mathbf{Z} \mid \Theta)=\sum_{i=1}^m\ln\sum_{z^{(i)}}P(x^{(i)},z^{(i)}|\Theta)\\&=\sum_{i=1}^m\ln\sum_{z^{(i)}}Q_i(z^{(i)})\frac{P(x^{(i)},z^{(i)}|\Theta)}{Q_i(z^{(i)})}\\&\geq\sum_{i=1}^m\sum_{z^{(i)}}Q_i(z^{(i)})\ln\frac{P(x^{(i)},z^{(i)}|\Theta)}{Q_i(z^{(i)})}.
\end{aligned}
$$

其中$Q_i$是隐变量$Z$的概率分布$Q_i(z^{(i)})=P(z^{(i)}|x^{(i)},\Theta)$，所以$\sum_{z^{(i)}}Q_i(z^{(i)})=1$.

最后的不等式是因为Jensen不等式，$\ln$是上凸函数{$f(E[X])\geq E[f(X)]$}。等号成立条件$\frac{P(x^{(i)},z^{(i)}|\Theta)}{Q_i(z^{(i)})}$为常数，即
$$
\frac{P(x^{(i)},z^{(i)}|\Theta)}{Q_i(z^{(i)})}=c.
$$
而
$$
\frac{P(x^{(i)},z^{(i)}|\Theta)}{Q_i(z^{(i)})}=\frac{P(x^{(i)},z^{(i)}|\Theta)}{P(z^{(i)}|x^{(i)},\Theta)}=P(x^{(i)}|\Theta),
$$

其在$\Theta$固定的条件下为常数，因此等号成立。

由此，EM算法可以改写为

- 设参数 $\Theta$ 已知，根据训练数据推断出最优隐变量 $\mathbf{Z}$ 的值 （$\mathrm{E}$步 ）：$Q_i(z^{(i)}):=P(z^{(i)}|x^{(i)},\Theta)$，$LL(\Theta \mid \mathbf{X},\mathbf{Z}):=\sum_{i=1}^m\sum_{z^{(i)}}Q_i(z^{(i)})\ln\frac{P(x^{(i)},z^{(i)}|\Theta)}{Q_i(z^{(i)})}$；
- 设 $\mathbf{Z}$ 的值已知，对参数 $\Theta$ 做极大似然估计（$\mathrm{M}$步 ) ：$\Theta:=\arg\max_\Theta LL(\Theta \mid \mathbf{X},\mathbf{Z})$.

EM算法应用广泛，如训练HMM（Baum-Welch算法），学习贝叶斯网络的概率，EM-聚类，学习单词对齐语言翻译，学习微信好友网络等。

**EM聚类**

$Q_i(z^{(i)})$为第$i\ (i\in\{1,...,m\})$个样本属于$z^{(i)}$的概率，设随机变量$\mathbf{X}$由$k$个高斯分布混合而成，则$z^{(i)}\in\{1,...,k\}$. 取各个高斯分布的概率为$\{\alpha_1,...,\alpha_k\}:=\alpha$，且$\sum_{i=1}^k\alpha_i=1$. 来自随机变量$\mathbf{X}$的$n$维样本$x\sim N(\mu,\Sigma)$. 

于是其概率密度函数为
$$
p(x)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\cdot\exp^{-\frac 1 2 (x-\mu)^\top\Sigma^{-1}(x-\mu)}.
$$
由此，隐变量 $\mathbf Z=\{z^{(1)},...,z^{(m)}\}$，其中$z^{(i)}\in\{1,...,k\}$表示样本$x^{(i)}$的高斯混合成分，参数$\Theta=\{\alpha,\Sigma,\mu\}$.

根据EM算法

- 设参数 $\Theta$ 已知，根据训练数据推断出最优隐变量 $\mathbf{Z}$ 的值 （$\mathrm{E}$步 ）：求样本$x^{(i)}$属于$z^{(i)}=j$的概率

$$
\gamma_j^i=Q_i(z^{(i)}=j)=P(z^{(i)}=j|x^{(i)},\alpha,\mu,\Sigma)=\frac{\alpha_j\cdot P(x^{(i)}|\mu_j,\Sigma_j)}{\sum_{l=1}^k\alpha_l\cdot P(x^{(i)}|\mu_l,\Sigma_l)}.
$$

参数的似然函数为
$$
LL(\alpha,\mu,\Sigma \mid \mathbf{X},\mathbf{Z})=\sum_{i=1}^m\sum_{z^{(i)}}Q_i(z^{(i)})\ln\frac{P(x^{(i)},z^{(i)}|\alpha,\mu,\Sigma)}{Q_i(z^{(i)})}.
$$

- 设 $\mathbf{Z}$ 的值已知，对参数 $\Theta$ 做极大似然估计（$\mathrm{M}$步 )：

$$
\{\alpha,\mu,\Sigma\}=\arg\max_{\{\alpha,\mu,\Sigma\}} LL(\alpha,\mu,\Sigma \mid \mathbf{X},\mathbf{Z}).
$$

最终参数更新可简化为$\mu_j:=\displaystyle\frac{\sum_{i=1}^m\gamma_{j}^ix_i}{\sum_{i=1}^m\gamma_{j}^i}$，$\Sigma_j:=\displaystyle\frac{\sum_{i=1}^m\gamma_j^i(x_i-\mu_j)(x_i-\mu_j)^\top}{\sum_{i=1}^m\gamma_j^i}$，$\alpha_j=\displaystyle\frac 1 m \sum_{i=1}^m\gamma^i_j$.

最后收敛后，进行簇划分，$x^{(i)}$属于簇$z^{(i)}=\arg\max_{j=1,...,k}\gamma_j^i$.

