# 合奏学习

基本想法：如果一个分类器运行良好，使用多个分类器共同作用。

采用多数s投票，则合奏学习的利益：对$m$个分类器，设$r$为一个分类器犯错误的概率，则
$$
p(\text { error })=\sum_{i=(m+1) / 2}^{m}\left(\begin{array}{c}
m \\
i
\end{array}\right) r^{i}(1-r)^{m-i}
$$
（二项分布的累积分布函数）

如何获得独立的分类器？

## 思路1：不同的学习方法

决策树，k-近邻，感知器，朴素贝叶斯，梯度下降变种1，梯度下降变种2……

优势：很多已有的方法，证明对一些问题表现较好

劣势：分类器经常不独立，犯同样的错误。比如这些都是线性模型，如果它们都犯同样错误，投票没有帮助。

## 思路2：分割训练数据

使用相同学习算法，但训练不同部分的训练数据。

优势：从不同数据学习，不会过拟合；容易实现；快速。

劣势：每个分类器只在小量的数据上训练，不一定比用全部数据加上正则化方法训练更好。

## 思路3：装袋（Bagging）

训练数据和原始训练数据大小一样，使用训练数据作为数据生成分布，通过**有重复抽样**生成训练数据。

通过取样从原始训练数据集中创建$m$个“新”训练数据集（称为$m$"bootstrap"样本）在每个数据集上训练分类器分类，从$m$个分类器中获得多数投票。

- **On average, how many of the original observations are not present in an average bootstrap sample?**

某个样本在$n$次抽样中任何一次都没被选中的概率为
$$
p=(1-1/n)^n.
$$
$n$变大时，
$$
\lim_{n\rightarrow \infty}(1-1/n)^n=1/e\approx 0.368,
$$
且收敛速度非常快，于是$p\approx 0.368$. 则有$n$个样本，每个样本不出现在bootstrap样本中的概率为$p$，于是不出现在bootstrap样本的样本数$X\sim B(n,p)$，即
$$
P(X=k)=f(k,n,p)=\left(\begin{array}{c}
n \\
k
\end{array}\right)p^k(1-p)^{n-k},
$$
则$E[X]=np$. 所以平均上，bootstrap样本包含了原样本的63.2%（$1-p$）。

- 什么时候装袋表现较好

装袋倾向于减少分类器的方差；通过投票，分类器对于有噪音的数据更加稳健。

装袋对于以下分类器有用：

1. 不稳定的分类器：训练集中的微小变化产生了非常不同的模型
2. 容易过度拟合的分类器，装袋有和正则化类似的作用

## 思路4：提升算法（Boosting）

- 强分类器

给定一定量的训练数据，一个目标误差率$\epsilon$，一个失败概率$p$，一个强分类器将以概率$1-p$产生一个误差率$<\epsilon$的分类器。

- 弱分类器

给定一定量的训练数据，一个失败概率$p$，一个弱分类器将以概率$1-p$产生一个误差率$<0.5$的分类器。

弱分类器更好构造，通过Boosting提升。

训练：开始给所有数据赋予相同权重，迭代中学习一个弱分类器并保存，然后改变数据权重。对正确分类的数据减小权重，对错误分类的数据增加权重，从而学习另外一个弱分类器。

分类：从每个学习到的分类器做预测，基于每个弱分类器的表现对预测做加权投票。

- Adaboost算法

Adaboost是Boosting中的一种，算法：

**输入**：训练集 $D=\left\{\left(\boldsymbol{x}_{1}, y_{1}\right), \cdots,\left(\boldsymbol{x}_{m}, y_{m}\right)\right\}$; 分类器算法 $\mathfrak{L} ;$ 训练轮数$T$;

**过程**：

(a)$\mathcal{D}_{1}(\boldsymbol{x})=1 / m$.
(b) 对 $t=1, \cdots, T$, 执行:
(c) $h_{t}=\mathfrak{L}\left(D, \mathcal{D}_{t}\right)$
(d) $\epsilon_{t}=P_{\boldsymbol{x} \sim \mathcal{D}_{t}}\left(h_{t}(\boldsymbol{x}) \neq f(\boldsymbol{x})\right)$;
(e) 如果 $\epsilon_{t}>0.5$, 则停止；否则，继续执行;
(f) $\alpha_{t}=\frac{1}{2} \ln \left(\frac{1-\epsilon_{t}}{\epsilon_{t}}\right)$;
(g) 令
$$
\begin{aligned}
\mathcal{D}_{t+1}(\boldsymbol{x}) &=\frac{\mathcal{D}_{f}(\boldsymbol{x}) \exp \left(-\alpha_{t} f(\boldsymbol{x}) h_{t}(\boldsymbol{x})\right)}{Z_{t}} \\
&=\frac{\mathcal{D}_{t}(\boldsymbol{x})}{Z_{t}} \times\left\{\begin{array}{ll}
\exp \left(-\alpha_{t}\right), & \text { 如果 } h_{t}(\boldsymbol{x})=f(\boldsymbol{x}) \\
\exp \left(\alpha_{t}\right), & \text { 如果 } h_{t}(\boldsymbol{x}) \neq f(\boldsymbol{x})
\end{array}\right.
\end{aligned}
$$
其中 $Z_{t}$ 是归一化常数;
(h) 循环结束.

**输出**： $H(\boldsymbol{x})=\operatorname{sign}\left(\sum_{t=1}^{T} \alpha_{t} h_{t}(\boldsymbol{x})\right)$.

从偏差-方差权衡角度，Adaboost更关注降低偏差。

- 提升决策树桩

最常用的分类器之一是决策树：可以使用浅（2-3级树），更常见的是1级树，叫决策树桩，询问有关单个特征的一个问题。

线性分类器，每个树桩定义该维度的权重，如果学习了那个维度的多个树桩，那么它就是加权平均值。

boosting在各种问题上应用非常成功，即使经过大量迭代也不会过度拟合。

