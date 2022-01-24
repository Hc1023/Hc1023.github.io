[Hc's blog](https://hc1023.github.io/)

[Hc's blog: R与统计学习基础](https://hc1023.github.io/2021/06/13/Statistical-learning/)

[机器学习与人工智能 浙江大学 中国大学MOOC(慕课)](https://www.icourse163.org/course/ZJU-1206689820)

[TOC]

# 正则化方法

## 基于模型的机器学习

1. 选择一个模型
   $$
   0=b+\sum_{j=1}^{m} w_{j} f_{j}
   $$

2. 选择一个优化标准（目标函数）
   $$
   \sum_{i=1}^{n} 1\left[y_{i}\left(w \cdot x_{i}+b\right) \leq 0\right]
   $$

3. 建立一个学习算法
   $$
   \operatorname{argmin}_{w, b} \sum_{i=1}^{n} 1\left[y_{i}\left(w \cdot x_{i}+b\right) \leq 0\right]
   $$

其中，为了让上述0/1损失函数有更好的数学性质，用一个凸的代理损失函数代替
$$
\sum_{i=1}^{n} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right),
$$
则学习算法变为
$$
\operatorname{argmin}_{w, b} \sum_{i=1}^{n} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right).
$$

## 梯度下降

选择一个初始点($w$)，重复直到损失函数不会在任何维度减小为止:
- 选择一个方向
- 在这个方向移动一小步，以减小损失函数 (使用梯度)
$$
w_{j}=w_{j}-\eta \frac{d}{d w_{j}} \operatorname{loss}(w)
$$

$$
\begin{aligned}
\frac{d}{d w_{j}} \text { loss } &=\frac{d}{d w_{j}} \sum_{i=1}^{n} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right) \\
&=\sum_{i=1}^{n} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right) \frac{d}{d w_{j}}-y_{i}\left(w \cdot x_{i}+b\right) \\
&=\sum_{i=1}^{n}-y_{i} x_{i j} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right)
\end{aligned}
$$

问题：优化在训练集上计算，可能过拟合，最小化训练集得到的$w,b$一般不会使测试集也最小。

## 偏差和方差

数据集和回归模型
$$
\mathcal{D}=\left\{\left(x_{1}, y_{1}\right), \ldots,\left(x_{N}, y_{N}\right)\right\} ; f(x ; \mathcal{D})
$$
用均方误差作为损失函数:
$$
\mathrm{MSE}=E\left[(y-f(x ; \mathcal{D}))^{2}\right]
$$
$\mathrm{MSE}=$ 偏差$^2$+方差

偏差：模型预测 ( 跨数据集 ) 与目标之间的平均差异。

方差：给定点的模型方差（跨数据集）。

- 偏差-方差平衡

![1622255735786](https://raw.githubusercontent.com/Hc1023/Hc1023.github.io/master/zone/statistical_learning/1622255735786.png)

增加训练数据可以允许使用更复杂的模型，即向右移动最优复杂度。

## p-范数正则化

正则化项是损失函数的附加标准，以确保不会过度拟合。
$$
\operatorname{argmin}_{w, b} \sum_{i=1}^{n} \operatorname{loss}\left(y y^{\prime}\right)+\lambda \text { regularizer }(w, b)
$$

对于模型$0=b+\sum_{j=1}^{m} w_{j} f_{j}$，我们希望权重不宜太大；无用的特征赋予0，权重为0.

如果能确保损失函数+正则化项是凸的，可以用梯度下降。

p-范数正则：

- 正则项：1-范数

$$
\operatorname{argmin}_{w, b} \sum_{i=1}^{n} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right)+\|w\|
$$

$$
\begin{aligned} \frac{d}{d w_{j}} \text { objective } &=\frac{d}{d w_{j}} \sum_{i=1}^{n} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right)+\lambda\|w\| \\ &=-\sum_{i=1}^{n} y_{i} x_{i j} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right)+\lambda \operatorname{sign}\left(w_{j}\right) \end{aligned}
$$

迭代
$$
w_{j}=w_{j}+\eta y_{i} x_{i j} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right)-\eta \lambda \operatorname{sign}\left(w_{j}\right)
$$

- 正则项：2-范数

$$
\begin{aligned} 
\frac{d}{d w_{j}} \text { objective }&=\frac{d}{d w_{j}} \sum_{i=1}^{n} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right)+\frac{\lambda}{2}\|w\|^{2}\\
&=-\sum_{i=1}^{n} y_{i} x_{i j} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right)+\lambda w_{j}
\end{aligned}
$$

迭代
$$
w_{j}=w_{j}+\eta y_{i} x_{i j} \exp \left(-y_{i}\left(w \cdot x_{i}+b\right)\right)-\eta \lambda w_{j}
$$

![1622257109162](https://raw.githubusercontent.com/Hc1023/Hc1023.github.io/master/zone/statistical_learning/1622257109162.png)

如果 $w_{j}$ 为正, 降低 $w_{j}$；如果 $w_j$ 为负, 增加 $w_j$.

- L1：$w_{j}=w_{j}+\eta\left(\operatorname{loss}_{-} \operatorname{correction}-\lambda \operatorname{sign}\left(w_{j}\right)\right)$

受欢迎，因为它往往导致稀疏解决方案（即大量零权重），但是，它不可导，因此仅适用于梯度下降求解法。

- L2：$w_{j}=w_{j}+\eta\left(\operatorname{loss}_{-} \operatorname{correction}-\lambda w_{j}\right)$

受欢迎，因为对于某些损失函数，可以直接求解（不需要梯度下降，但通常仍然需要迭代求解）。

- Lp：$w_{j}=w_{j}+\eta\left(\operatorname{loss}_{-} \operatorname{correction}-\lambda cw_{j}^{p-1}\right)$

不太受欢迎，因为对权重缩减不够。

scikit学习包：http://scikit-learn.org/stable/modules/sgd.html

### 通用名称

- （普通）最小二乘：平方损失
- 岭回归：L2正则化的平方损失
- Lasso回归：L1正则化的平方损失
- 弹性（Elastic）回归：L1和L2正则化平方损失的组合
- 逻辑斯蒂回归：logistic损失