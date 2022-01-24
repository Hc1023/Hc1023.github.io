[Hc's blog](https://hc1023.github.io/)

[Hc's blog: R与统计学习基础](https://hc1023.github.io/2021/06/13/Statistical-learning/)

[机器学习与人工智能 浙江大学 中国大学MOOC(慕课)](https://www.icourse163.org/course/ZJU-1206689820)

[TOC]

# 人工神经网络（ANN）

神经网络：

- 针对难以用准则来描述的复杂模型
- 能够达到更深层次的抽象
- 能够有广泛使用的分类算法

特点：

- 能充分逼近复杂的非线性关系
- 可以并行分布处理
- 高度的鲁棒性和容错能力
- 学习能力和自组织能力

>  1943年，McCulloch和Pitts将生物神经细胞抽象成一个简单的数学模型，即经典的M-P神经元模型。

神经元模型：输入$\boldsymbol{x}$、输出$h_{\boldsymbol{w}, b}(\boldsymbol{x})$、权重$\boldsymbol{w}$、偏置$b$、激活函数$f$
$$
h_{\boldsymbol{w}, b}(\boldsymbol{x})=f\left(\sum_{i=1}^{3} w_{i} x_{i}+b\right)=f(z)
$$
$f$最简单的形式为阶跃函数，
$$
f(z)=\left\{\begin{array}{ll}
1 & z \geq 0, \\
0 & z<0,
\end{array}\right.
$$
用以表示兴奋或抑制状态。但是由于其不连续、不光滑等不好的数学性质，实际应用中不予采用，会使用光滑化后的近似函数。

## 感知器

> 感知器（perceptron）的结构类似于神经元模型，是对神经元最基本概念的模拟。它是由Rosenblatt在1957年首次提出，由两层神经元构成，即输入层和输出层。

感知器可以解决与、或、非，还可以解决线性分类、线性回归问题。

学习权重$\boldsymbol{w}$和偏置$b$使得以下代价函数（cost function）/损失函数（loss function）最小：
$$
\frac{1}{2 n} \sum_{k=1}^{n}\left(\hat{y}_{(k)}-h_{\boldsymbol{w}, b}\left(x_{(k)}\right)\right)^{2}
$$

- 感知器的学习算法

对于训练数据 $(\boldsymbol{x}, \hat{y})$， $\boldsymbol{w}$ 和 $\mathrm{b}$ 进行如下的更新：
$$
\begin{aligned} \boldsymbol{w}^{\text {new }} &=\boldsymbol{w}^{\text {old }}+\eta \Delta \boldsymbol{w}^{\text {old }}, \\ \Delta \boldsymbol{w}^{\text {old }} &=(\hat{y}-y) x, \\ b^{\text {new }} &=b^{\text {old }}+\eta \Delta b^{\text {old }}, \\ \Delta b^{\text {old }} &=\hat{y}-y, \end{aligned}
$$
其中 $\eta$ 是学习率（梯度下降法的搜索步长）。

单层感知器只能对线性可分的数据集进行分类，不能解决逻辑问题异或（XOR）。

多层感知器可以模拟任意复杂的函数，其中函数的复杂性决定了网络的神经元个数和隐层层数。

## 神经网络模型

多层感知器也叫神经网络，也即人工神经网络，它除了输入层、输出层，中间可以有一至多个隐层。

- 前馈式神经网络

每个神经元只与前一层的神经元相连。当前层，只接收前一层的输出，而自身的输出只能输出给下一层，各层间没有反馈。注意偏置不能有输入。

- 反馈神经网络

每个神经元同时将自身的输出作为输入信号反馈给其它神经元。网络的连接图中具有回路，所以反馈神经网络需要工作一段时间后才能达到稳定。Hopfield神经网络是最简单且应用广泛的反馈神经网络。

为叙述方便，引入如下记号：

- $n_{l}:$ 表示神经网络的层数 。
- $s_{l}:$ 表示第1层的单元数量(不包括偏置单元)。
- $W^{(l)}:$ 表示第 $l$ 层到 $l+1$ 层的权重矩阵。其中 $W_{i j}^{(l)}$ 是第$l$层第$j$单元到 $l+1$ 层第 $i$ 单元之间的权重。
- $b^{(l)}:$ 表示第 $l+1$ 层的偏置，其中 $b_{i}^{(l)}$ 第 $l+1$ 层的第$i$单元的偏置项。
- $z^{(l)}$ :表示第$l$层的每个单元的输入加权和，其中 $z_{i}^{(l)}$ 表示第$l$层第 $i$ 单元的输入加权和。
- $a^{(l)}:$ 表示第 $l$ 层的激活值（输出值），其中 $a_{i}^{(l)}$ 表示第$l$层第 $i$ 单元的激活值。

例如：
$$
\begin{aligned}
&z_{1}^{(2)}=W_{11}^{(1)} x_{1}+W_{12}^{(1)} x_{2}+W_{13}^{(1)} x_{3}+b_{1}^{(1)} \\
&a_{1}^{(2)}=f\left(z_{1}^{(2)}\right), \\
&z_{2}^{(2)}=W_{21}^{(1)} x_{1}+W_{22}^{(1)} x_{2}+W_{23}^{(1)} x_{3}+b_{2}^{(1)}, \\
&a_{2}^{(2)}=f\left(z_{2}^{(2)}\right), \\
&z_{3}^{(2)}=W_{31}^{(1)} x_{1}+W_{32}^{(1)} x_{2}+W_{33}^{(1)} x_{3}+b_{3}^{(1)}, \\
&a_{3}^{(2)}=f\left(z_{3}^{(2)}\right)
\end{aligned}
$$
输出为 $$h_{W, b}(x)=a^{(3)}=f\left(z^{(3)}\right)=f\left(W_{11}^{(2)} a_{1}^{(2)}+W_{12}^{(2)} a_{2}^{(2)}+W_{13}^{(2)} a_{3}^{(2)}+b_{1}^{(2)}\right).$$

用向量表示各层的输入和输出，并定义 $f\left(\left(z_{1}, z_{2}, z_{3}\right)^{\top}\right)=\left(f(z_{1}), f(z_{2}),f(z_{3})\right)^{\top}$, 则上式可简化为:
$$
\begin{aligned}
z^{(2)} &=W^{(1)} x+b^{(1)} \\
a^{(2)} &=f\left(z^{(2)}\right) \\
z^{(3)} &=W^{(2)} a^{(2)}+b^{(2)} \\
h_{W, b}(x) &=a^{(3)}=f\left(z^{(3)}\right)
\end{aligned}
$$
神经网络模型：定义代价函数，根据样本数据和标签采用梯度下降法进行学习，求得权重等参数，对测试数据进行预测。

涉及：激活函数，代价函数，梯度下降法，反向传播算法。

## 激活函数和代价函数

激活函数：使模型具有**非线性**性

- Sigmoid函数：$f(x)=\frac{1}{1+e^{-x}}$

但是具有**饱和性**，$\lim _{x \rightarrow \infty} f^{\prime}(x)=0$，参数更新中，代价函数关于参数的偏导数会包含激活函数的导数项$f^{\prime}(x)$。当范围落入饱和区，会使得参数难以更新。还会导致向下一层传递的梯度也非常小，这种现象称为**梯度消失**，会使得整个网络的参数很难得到有效的训练。

- Tanh函数（双曲正切函数）：$f(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}=\frac{1-e^{-2 x}}{1+e^{-2 x}}$

范围在$(-1,1)$，其收敛速度快于Sigmoid函数，但同样具有饱和性，会造成梯度消失。

- ReLU函数：$f(x)=\left\{\begin{array}{ll}x, & \text { 当 } x \geq 0, \\ 0, & \text { 当 } x<0 .\end{array}\right.$

线性整流单元（rectified linear units），在正半轴上不存在梯度衰减问题，但在小于0区域，同样对应权重无法更新，称这种现象为**神经元坏死**。由于神经元坏死会影响网络的收敛性，因此提出了各种ReLU函数的变形。

- Softplus函数：$f(x)=\ln \left(1+e^{x}\right)$

它是ReLU函数的近似。

代价函数：训练模型使代价函数达到最小

- 均方误差

$$
J(\theta)=\frac{1}{n} \sum_{x} L(x, y ; \theta)=\frac{1}{2 n} \sum_{x}\|y-\hat{y}\|^{2}
$$
$\theta$ 为所求的参数集 （权值$W$ 和偏置$b$），$n$ 是样本总数， $y$ 是训练数据的标签， $\hat{y}=h_{W, b}(x)$ 是 $x$ 的神经网络输出。

均方误差作为代价函数的优点是形式简单，通俗易懂，但由于其修正项$\frac{\partial J}{\partial w_{j}}$包含激活函数的导数，容易出现饱和现象，容易陷于局部极值。由此可以用交叉熵改进。

- 交叉熵

$$
J(\theta)=-\frac{1}{n} \sum_{x}[y \ln \hat{y}+(1-y) \ln (1-\hat{y})]
$$

熵是由Shannon最早引入信息论，用来衡量不确定性程度；变量不确定性越大，熵越大。

例如，考虑激活函数为Sigmoid函数的情况：

考虑最外层（输出层），神经网络的输出
$$
\hat y=a^{(N)}=\sigma\left(z^{(N)}\right)=\sigma\left(\sum w_{j}^{(N-1)} a_{j}^{(N-1)}+b^{(N-1)}\right).
$$
利用$\sigma^{\prime}(z)=\sigma(z)(1-\sigma(z))$，则
$$
\begin{aligned}
\frac{\partial J}{\partial w_{j}} &=-\frac{1}{n} \sum_{x}\left(\frac{y}{\sigma\left(z^{(N)}\right)}-\frac{1-y}{1-\sigma\left(z^{(N)}\right)}\right) \frac{\partial \sigma}{\partial w_{j}} \\
&=\frac{1}{n} \sum_{x} \frac{\sigma^{\prime}\left(z^{(N)}\right) a_{j}^{(N-1)}}{\sigma\left(z^{(N)}\right)\left(1-\sigma\left(z^{(N)}\right)\right)}\left(\sigma\left(z^{(N)}\right)-y\right) \\
&=\frac{1}{n} \sum_{x}\left(\sigma\left(z^{(N)}\right)-y\right) a_{j}^{(N-1)}.
\end{aligned}
$$
可以看出修正项$\frac{\partial J}{\partial w_{j}}$不再出现Sigmoid激活函数的导数项$\sigma^{\prime}(z)$，而留下了误差项$\sigma\left(z^{(N)}\right)-\hat{y}$，如果误差越大，修正项就越大，参数更新就越快，训练速度也就越快。

## 梯度下降法和反向传播法

- 神经网络模型学习的目的是找到能最小化代价函数的参数（权重和偏置）。

- 为了解决这一优化问题，通常采用梯度下降法，又称批量梯度下降法，也称为最速下降法。

$$
\theta^{\text {new }}=\theta^{\text {old }}-\alpha \nabla_{\theta} J\left(\theta^{\text {old }}\right)
$$

其中$\alpha$称为步长，在机器学习中通常称为学习率。神经网络模型的学习常常需要大的训练集来得到较号的泛化能力。但由公式可知梯度下降法需要在每一迭代中对每个样本进行计算
$$
\nabla_{\theta} J=\frac{1}{n} \sum_{x} \nabla_{\theta} L(x, y; \theta).
$$
对每一个样本求梯度，对所有样本求平均，显然当训练集样本量较大时，会使得训练消耗相当长的时间。甚至使得整个算法的整体计算量过大，时间过长，而被放弃。因此在实际计算中，往往采用小规模样本的梯度期望，或仅根据一个样本的梯度来近似代替。

- 随机梯度下降算法（SGD）
- 小批量梯度下降算法（MBGD）

具体而言，从训练集中随机抽出一小批量样本$\{x(1),...,x(m)\}$，对应的标签为$\{y(1),...,y(m)\}$。那么，代价函数的梯度近似为所抽取样本的梯度期望
$$
\nabla J(\theta) \approx \frac{1}{m} \sum_{k=1}^{m} \nabla_{\theta} L\left(x_{(k)}, y_{(k)} ; \theta\right).
$$
其中的求和只对随机抽样出来的样本进行。小批量梯度下降法在迭代每一步，参数的更新只用了小批量样本的信息。相对于整个样本数据集来说，小批量样本的数目$m$相对于样本总数$n$要小得多，计算量大大减少，计算速度也可以极大地提升。

训练过程中，将遍历一次所有样本的计算称为一个epoch，在梯度下降法中迭代一次就是一个epoch，而在小批量梯度下降算法中，迭代（总样本数/批尺寸）次才是一个epoch，小批量梯度下降法是以迭代次数换取算法的运行速度。随机梯度下降算法是小批量梯度下降法的一个特里，即$m=1$的情况。小批量梯度下降法和随机梯度下降法虽然每一次迭代的计算量大大减少，但迭代方向可能不是收敛方向（剧烈震荡），算法收敛可能需要更多的迭代次数。

- 小批量梯度下降算法：

1: 给定参数初始值$\theta$，学习率 $\alpha$；

2: while 终止条件未满足 do

3: 	从训练集中随机抽出 $m$ 个样本 $\left\{x_{(1)}, \cdots, x_{(m)}\right\}$, 相应的标签为$\left\{y_{(1)}, \cdots, y_{(m)}\right\}$

4: 	计算小批量样本的梯度
$$
\Delta \theta=\frac{1}{m} \sum_{k=1}^{m} \nabla_{\theta} L\left(x_{(k)}, y_{(k)} ; \theta\right)
$$
5: 	参数更新 $\theta \rightarrow \theta-\alpha \Delta \theta$.

6: end while

考虑网络的传播过程

$$
X\rightarrow z^{(2)}\rightarrow a^{(2)}\rightarrow z^{(3)}\rightarrow \ldots \ldots\rightarrow z^{(N)}\rightarrow a^{(N)}=Y\rightarrow J(\theta).
$$

可知输出的迭代形式

$$
\begin{aligned}
Y &=a^{(N)}=\sigma\left(z^{(N)}\right)=\sigma \circ f_{N}\left(a^{(N-1)}\right) \\
&=\sigma \circ f_{N} \circ \sigma\left(z^{(N-1)}\right) \\
&=\cdots \cdots \\
&=\sigma \circ f_{N} \circ \cdots \circ \sigma \circ f_{2}(X),
\end{aligned}
$$
其中$\quad f_{i}\left(z^{(i)}\right)=W^{(i-1)} a^{(i-1)}+b^{(i-1)}$，$\theta^{\text {new }}=\theta^{\text {old }}-\alpha \nabla_{\theta} J\left(\theta^{\text {old }}\right)$。由此迭代更新参数。

而偏导数值根据反向传播和链式法则可知
$$
\frac{\partial J}{\partial W_{i j}^{(l)}}=\frac{\partial J}{\partial z_{i}^{(l+1)}} \frac{\partial z_{i}^{(l+1)}}{\partial W_{i j}^{(1)}},
$$
由$z_{i}^{(l+1)}=\sum_{j=1}^{s_{l}} W_{i j}^{(l)} a_{j}^{(l)}+b_{i}^{(l)}$，得到
$$
\frac{\partial z_{i}^{(l+1)}}{\partial W_{i j}^{(l)}}=\frac{\sum_{k=1}^{s_{l}} W_{i k}^{(l)} a_{k}^{(l)}+b_{i}^{(l)}}{\partial W_{i j}^{(l)}}=a_{j}^{(l)}.
$$
记$\delta_{i}^{(l)}=\frac{\partial J(W, b ; x, y)}{\partial z_{i}^{(l)}}$，所以
$$
\frac{\partial J}{\partial W_{i j}^{(l)}}=\delta_{i}^{(l+1)} a_{j}^{(l)},\  \frac{\partial J}{\partial b_{i}^{(l)}}=\delta_{i}^{(l+1)}.
$$
$a_{j}^{(l)}$可由输入层开始正向计算，而$\delta_{i}^{(l+1)}$需要反向计算。

下面我们只考虑一个固定训练样本$x$，代价函数为均方误差的情况。

引入哈达玛（Hadamard）乘积：$(A \odot B)_{i j}=(A)_{i j}(B)_{i j}$

先计算输出层的误差项
$$
\begin{aligned}
\delta_{i}^{(N)}=& \frac{\partial J}{\partial z_{i}^{(N)}}=\frac{1}{2} \frac{\partial}{\partial z_{i}^{(N)}}\|y-\hat y\|^{2} \\
=& \frac{1}{2} \sum_{j=1}^{s_{N}} \frac{\partial}{\partial z_{i}^{(N)}}\left(y_{j}-a_{j}^{(N)}\right)^{2} \\
=&\left(a_{i}^{(N)}-y_{i}\right) \sigma^{\prime}\left(z_{i}^{(N)}\right),
\end{aligned}
$$
写成矩阵向量的形式为
$$
\delta^{(N)}=\left(a^{(N)}-y\right) \odot \sigma^{\prime}\left(z^{(N)}\right).
$$
然后计算中间第$l$隐层的误差项。根据链式法则，有
$$
\begin{aligned}
\delta_{i}^{(l)} &=\frac{\partial J}{\partial z_{i}^{(l)}}=\sum_{k=1}^{s_{l+1}} \frac{\partial J}{\partial z_{k}^{(l+1)}} \frac{\partial z_{k}^{(l+1)}}{\partial z_{i}^{(l)}} \\
&=\sum_{k=1}^{s_{l+1}} \delta_{k}^{(l+1)} \frac{\partial z_{k}^{(l+1)}}{\partial z_{i}^{(l)}}, \quad l=N-1, \cdots, 2.
\end{aligned}
$$
再由
$$
z_{k}^{(l+1)}=\sum_{j=1}^{s_{l}} W_{k j}^{(l)} a_{j}^{(l)}+b_{k}^{(l)}=\sum_{j=1}^{s_{l}} W_{k j}^{(l)} \sigma\left(z_{j}^{(l)}\right)+b_{k}^{(l)},
$$
故 
$$
\delta_{i}^{(l)}=\sum_{k=1}^{s_{l+1}} \delta_{k}^{(l+1)} W_{k i}^{(l)} \sigma^{\prime}\left(z_{i}^{(l)}\right),
$$
即
$$
\quad \delta^{(l)}=\left(W^{(l)}\right)^{T} \delta^{(l+1)} \odot \sigma^{\prime}\left(z^{(l)}\right).
$$

- 反向传播算法：

1: 根据输入的训练数据$x$，计算$a^{(1)}$；

2: 根据神经网络正向计算
$$
z^{(l)}=W^{(l-1)} a^{(l-1)}+b^{(l-1)}, \quad a^{(l)}=f\left(z^{(l)}\right), \quad l=2, \cdots, N;
$$
3: 计算输出层的误差项
$$
\delta^{(N)}=\left(a^{(N)}-\hat{y}\right) \odot f^{\prime}\left(z^{(N)}\right);
$$
4 : 反向计算
$$
\delta^{(l)}=\left(W^{(l)}\right)^{T} \delta^{(l+1)} \odot f^{\prime}\left(z^{(l)}\right), \quad l=N-1, \cdots, 2 ;
$$
5: 让算输出的偏导数
$$
\frac{\partial J}{\partial W_{i j}^{(l)}}=\delta_{i}^{(l+1)} a_{j}^{(l)}, \quad \frac{\partial J}{\partial b_{i}^{(l)}}=\delta_{i}^{(l+1)}.
$$