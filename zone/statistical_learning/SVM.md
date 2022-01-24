[Hc's blog](https://hc1023.github.io/)

[Hc's blog: R与统计学习基础](https://hc1023.github.io/2021/06/13/Statistical-learning/)

[机器学习与人工智能 浙江大学 中国大学MOOC(慕课)](https://www.icourse163.org/course/ZJU-1206689820)

[TOC]

# 支持向量机

## 最大间隔分类器

$p$维空间中，超平面$p-1$维平面放射子空间（意思是无需经过原点）$b^{\top} x+a=0$，记超平面$(b,a)$

样本空间任意点$x=(x_1,...,x_p)^{\top}$到超平面$(b,a)$的距离$d=\frac{|b^\top x + a|}{||b||}$.

根据$b^\top +a$的符号可判断样本落在超平面哪一侧。

**最大间隔超平面**是离训练观测最远的分割超平面。解决如下优化问题：
$$
\left\{\begin{array}{l}\min _{\boldsymbol{b}, a} \frac{1}{2}\|\boldsymbol{b}\|^{2} \\ \text { s.t. } y_{i}\left(\boldsymbol{b}^{\mathrm{T}} \boldsymbol{x}_{i}+a\right) \geq 1, i=1, \cdots, m\end{array}\right.
$$

决定最大间隔超平面的向量为支持向量。

 ## 支持向量分类器

分割超平面不存在，非完美分类的超平面分类器，允许小部分观测误分。

优化目标：

$$
\min _{b, a}\left\{\frac{1}{2}\|\boldsymbol{b}\|^{2}+C \sum_{i=1}^{m} l_{0 / 1}\left(y_{i}\left(\boldsymbol{b}^{\mathrm{T}} \boldsymbol{x}_{i}+a\right)-1\right)\right\}
$$

其中$C>0$是调节参数，在间隔大小和越过间隔面的观测个数之间维持权衡关系，$l_{0/1}$是“0/1损失函数”。由于$l_{0/1}$非凸、非连续，使用替代损失函数（凸的连续函数且是上界）。常用hinge损失函数：

$$
l_{\text {hinge }}(z)=\max (0,1-z)
$$

采用hinge损失，式可演化为:
$$
\min _{b, a}\left\{\frac{1}{2}\|b\|^{2}+C \sum_{i=1}^{m} \max \left(0,1-y_{i}\left(\boldsymbol{b}^{\mathrm{T}} \boldsymbol{x}_{i}+a\right)\right)\right\} .
$$
引入松他变量 $\xi_{i} \geqslant 0$, 将上式写为
$$
\left\{\begin{array}{c}
\min _{b, a, \xi_{i}}\left\{\frac{1}{2}\|b\|^{2}+C \sum_{i=1}^{m} \xi_{i}\right\} \\
\text { s.t. } y_{i}\left(\boldsymbol{b}^{\mathrm{T}} \boldsymbol{x}_{i}+a\right) \geq 1-\xi_{i} . \\
\xi_{i} \geq 0, i=1, \cdots, m .
\end{array}\right.
$$
松他变量表征该观测不满足约束式的程度.

支持向量分类器的判断规则只由**训练观测的一部分（支持向量）**确定，这意味着对于超平面较远的观测，分类器稳健。

区别于线性判别分析，取决于组内观测的均值，以及组内协方差矩阵。

## 支持向量机

非线性分类边界，使用核函数扩大特征空间。

对于线性不可分的分类问题，将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间里是线性可分的。

令 $\phi(\boldsymbol{x})$ 表示将$\boldsymbol{x}$映射后的特征向量，于是在特征空间中分割超平面所对应的模型可表示为:
$$
f(\boldsymbol{x})=\boldsymbol{b}^{\top} \phi(\boldsymbol{x})+a .
$$
接下来，求解如下的优化问题:
$$
\min _{b, a}\left\{\frac{1}{2}\|b\|^{2}+C \sum_{i=1}^{m} l_{0 / 1}\left(y_{i}\left(b^{\top} \phi\left(\boldsymbol{x}_{i}\right)+a\right)-1\right)\right\} .
$$
通常用hinge损失替代上式中 $l_{0 / 1 \circ}$ 此时，上式变成:
$$
\min _{b, a}\left\{\frac{1}{2}\|b\|^{2}+C \sum_{i=1}^{m} \max \left(0,1-y_{i}\left(b^{\top} \phi\left(\boldsymbol{x}_{i}\right)+a\right)\right)\right\} .
$$
引入松弛变量 $\xi_{\mathrm{i}} \geqslant 0$ 后，优化问题可写成:
$$
\left\{\begin{array}{l}
\min _{b, a, \xi_{i}}\left\{\frac{1}{2}\|b\|^{2}+C \sum_{i=1}^{m} \xi_{i}\right\} \\
\text { s.t. } y_{i}\left(b^{\top} \phi\left(x_{i}\right)+a\right) \geq 1-\xi_{i}, \\
\xi_{i} \geq 0, i=1, \cdots, m .
\end{array}\right.
$$


上述的优化问题最终转化为它的对偶问题来解决:
$$
\left\{\begin{array}{l}
\min _{\alpha}\left\{\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \phi^{\top}\left(\boldsymbol{x}_{i}\right) \phi\left(\boldsymbol{x}_{j}\right)\right\} \\
\text { s.t. } \sum_{i=1}^{m} \alpha_{i} y_{i}=0, \\
0 \leq \alpha_{i} \leq C, i=1, \cdots, m .
\end{array}\right.
$$
计算 $\phi^{\top}\left(x_{i}\right) \phi\left(x_{j}\right)$, 这是 $x_{i}$ 与 $x_{j}$ 映射到特征空间之后的内积，不易计算。引入核函数

$$
\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\phi^{\mathrm{T}}\left(\boldsymbol{x}_{i}\right) \phi\left(\boldsymbol{x}_{j}\right)
$$

对偶问题可重写为：
$$
\left\{\begin{array}{l}
\min _{\alpha}\left\{\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)\right\} \\
\text { s.t. } \sum_{i=1}^{m} \alpha_{i} y_{i}=0 . \\
0 \leq \alpha_{i} \leq C . i=1, \cdots, m
\end{array}\right.
$$
求解后即可得到:
$$
f(\boldsymbol{x})=\boldsymbol{b}^{\top} \phi(\boldsymbol{x})+a=\sum_{i=1}^{m} \alpha_{i} y_{i} \kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)+a
$$
| 核函数     | 表达式                                                       |
| ---------- | ------------------------------------------------------------ |
| 线性核     | $\quad \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\boldsymbol{x}_{i}^{\top} \boldsymbol{x}_{j}$ |
| 多项式核   | $\quad \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\left(\boldsymbol{x}_{i}^{\top} \boldsymbol{x}_{j}\right)^{d} \quad d \geq 1$ |
| 径向核     | $\quad \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\exp \left(-\gamma\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2}\right) \quad \gamma>0$ |
| 拉普拉斯核 | $\quad \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\exp \left(-\frac{\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|}{\sigma}\right) \quad \sigma>0$ |
| Sigmoid 核 | $\quad \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\tanh \left(\beta \boldsymbol{x}_{i}^{\top} \boldsymbol{x}_{j}+\theta\right) \quad \beta>0, \theta<0$ |


## code

`svm()`