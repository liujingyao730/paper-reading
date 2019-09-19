# GCN相关的基础文献
图卷积网络是针对于图结构数据的有力分析工具，这四篇文章是图卷积网络领域内比较重要的基础性工作，对于我们理解和使用图卷积网络有着很重要的意义
## 目录
* [before deep learing graph neural network](###图卷积)
* [convolutional neural networks on graphs with fast localized spectral filtering](#图卷积的快速计算)
* [Semi-supervised classification with graph convolutional network-iclr2017](#一阶近似与半监督学习)
* [Deeper insights into graph convolutional networks for semi-supervised learning aaai2018](#deeper-insights)
* [Diffusion convolutional recurrent neural network Data-driven traffic forecasting](#扩散卷积)
### 图卷积
在Euclidean domains(图像，声音信号等类型)的数据中，卷积操作就是将函数进行傅里叶变换映射到频域空间，之后进行相乘再做逆傅里叶变换得到的结果。对于图结构的数据，如果我们想要将卷积领域进行扩展，就需要合理的定义在图领域的傅里叶变换，进而可以定义图领域的卷积操作。  
把Euclidean domains中的傅里叶变换迁移拓展到图领域中，最核心的步骤在于把拉普拉斯算子的特征函数$e^{i\omega t}$在图领域中做出对应的映射，而这个对应的映射就是图拉普拉斯矩阵的特征向量
* 传统领域的傅里叶变换  <div id='label'>
  
> 传统的傅里叶变换定义为：$F(\omega)=\mathcal{F}[f(t)]=\int f(t)e^{i\omega t}dt$，这是信号$f(t)$与基函数$e^{i\omega t}dt$之间的积分，而基函数选择它的原因在于$e^{i\omega t}dt$是拉普拉斯算子的特征函数  
> 同样的，当我们想将卷积拓展到图领域时，因为图的拉普拉斯矩阵就是离散的拉普拉斯算子，所对应的选择的基函数就应当是图拉普拉斯矩阵的特征向量

* 图拉普拉斯矩阵的定义与分解

> 图的拉普拉斯矩阵通常定义为$L=D-A，$其中$D$指顶点度数组成的对角矩阵，$A$指邻接矩阵或者边权重矩阵；在运算中通常采用归一化后的拉普拉斯矩阵$L^{sys} = D^{-1/2}LD^{-1/2}$
> 图拉普拉斯矩阵是对称半正定矩阵，它的特征向量之间相互正交，所有的特征向量构成的矩阵成为正交矩阵，因此我们可以知道拉普拉斯矩阵一定可以进行谱分解，并且分解后有特殊的形式：
> $$L=U \left[ \begin{matrix}\lambda_1 & & \\ & ... & \\ & & \lambda_n\end{matrix}\right]U^T$$
> 其中$U$是由拉普拉斯矩阵特征向量组成的矩阵，而$\lambda$代表着拉普拉斯矩阵的特征值

* 图领域的傅里叶变换
> 仿照传统领域下的傅里叶变换定义，我们就可以得到图领域的傅里叶变换
> $$ F(\lambda_l)=\hat{f}(\lambda_l)=\sum^N_{i=1}f(i)u^*_l(i)$$
> $f$是图上的N维向量，$f(i)$表示节点i上对应的输入，$u_l(i)$表示第$l$个特征向量的第$i$个分量，特征值就对应了在不同基函数下对应的分量，也可以在一定程度上认为是对应的频率。$f$的图傅里叶变换就是与$\lambda_l$对应的特征向量$u_l$进行内积运算
> 进一步的，我们图傅里叶变换的矩阵形式写成：
> $$ \left[ \begin{matrix} 
> \hat f(\lambda_1) \\
> \hat f(\lambda_2) \\
> \vdots\\
> \hat f(\lambda_N) 
> \end{matrix} \right] = 
> \left[ \begin{matrix} 
> u_1(1) & u_1(2) & \cdots & u_1(N) \\
> u_2(2) & u_2(2) & \cdots & u_N(2) \\
> \vdots & \vdots & \ddots & \vdots \\
> u_1(N) & u_2(N) & \cdots & u_N(N)
> \end{matrix} \right]
> \left[\begin{matrix}
> \hat f(\lambda_1) \\
> \hat f(\lambda_2) \\
> \vdots\\
> \hat f(\lambda_N)
> \end{matrix}\right] $$
> 也就是说$\hat{f}=U^Tf$，逆傅里叶变换也可同样的形式推广：$f=U\hat{f}$

* 图卷积
> 对于卷积核$h$，我们将其进行傅里叶变换之后的结果写成对角矩阵的形式就是：$g(\Lambda)=diag(\hat{h}(\lambda_1), \hat{h}(\lambda_2),...,\hat{h}(\lambda_N))$，所以$h与f$的卷积就可以写成：
> $$f*_gh= Ug(\Lambda)U^Tf$$
### 图卷积的快速计算
回顾传统二维图像中的卷积，我们可以发现二维的卷积具备着两个很好的特性：
* 局部连接：每一个卷积核在不同的位置只对这一个位置的局部具备感受的能力，不接收其他区域的信息
* 参数共享：每一个卷积核在不同的位置都使用相同的参数，这就极大的减少了所需学习的参数数量

但是在上述图卷积的操作中这两点都不具备，每一个卷积核所需要学习的参数$g(\Lambda)=diag(\hat{h}(\lambda_1), \hat{h}(\lambda_2),...,\hat{h}(\lambda_N))$的规模为$\mathcal{O}(N)$，$N$表示节点数目，在多通道，多卷积核的情况下参数的数目相当的庞大；另外在这样的操作下就意味着每一个节点都可以看到所有的节点的信息，这样就不具备局部连接的特性  
因此，为了适应深度学习的需求，学者们对图卷积做了一定程度的改变使得它具备了局部感知特性并且降低了参数数量：
#### Polynomial parametrization for localized filters  
多项式参数化就是将原本作为卷积核参数的对角矩阵，由简单的学习对角矩阵对角线上每一个元素改变成学习一个多项式的系数，即：
$$g(\Lambda)=diag(\theta_1, \theta_2,...,\theta_N) \rightarrow g(\Lambda)=\sum_i^{K-1}\theta_i\Lambda^k$$
这样原本的卷积操作就变成：
$$\begin{aligned}
    f *_g h =& Ug(\Lambda)U^Tf\\
    =&U(\sum_i^{K-1}\theta_i\Lambda^k)U^Tf\\
    =&(\sum_i^{K-1}\theta_iL^k)f
\end{aligned}$$
这样就将原本的参数量从$\mathcal{O}(N)$降低到了$\mathcal{O}(K)$，同时由于拉普拉斯矩阵的特殊性，$K$具备着明确的含义，即每一个节点所能看到的节点的最近距离，这样就达到了降低参数量同时使得卷积核具备了局部连接性的目的
#### Recursive formulation for fast filtering
在进行了上述的变化之后，虽然降低了整体的参数量也避免了拉普拉斯矩阵的特征值分解计算，但在实际的计算中计算代价仍然不小（拉普拉斯矩阵的分解步骤可以在训练前就分解完成，整体更新参数时只要调用分解完成的特征向量矩阵即可，而同时因为需要计算拉普拉斯矩阵的乘积，空间存储本身的复杂度就在$\mathcal{O}(N^2)$级别，所以整体避免分解只是略微降低了空间复杂度）  
因此，为了避免$\mathcal{O}(N^3)$的矩阵乘（虽然在算法层面有更快级别的算法），作者提出采用切比雪夫多项式逼近的方法来通过递归计算拉普拉斯矩阵乘向量的方式降低计算的复杂度，即：
$$g(\Lambda)=\sum_i^{K-1}\theta_i\Lambda^i\approx\sum_i^{K-1}\theta_iT_i(\tilde{L})$$
其中$\tilde{L}=2L/\lambda_{max} - I_N$这是为了保证切比雪夫多项式的数学性质而做出的变换，切比雪夫多项式的计算公式为：$T_k(L)=2xT_{k-1}(x)-T_{k-1}(x), T_1(x)=x, T_0(x)=1$，这样，如果我们将$T_i(\tilde{L})f$记做$x_k$的话，卷积操作就变成$f *_g h=\sum_i^{K-1}x_i$，而每一个$x_i$可以通过$x_i = 2\tilde{L}x_{i-1}-x_{i-2}$的方式递归计算得到，而由于拉普拉斯矩阵是一个稀疏矩阵，那么整体的计算复杂度就降低到$\mathcal{O}(\mathcal{E})$，$\mathcal{E}$表示边的数量，这样就大大的降低的计算的复杂度，使得图卷积操作适应了深度学习的要求
### 一阶近似与半监督学习
### deeper-insights
### 扩散卷积