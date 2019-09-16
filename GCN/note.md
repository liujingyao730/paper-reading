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
> \[L=U \left[ \begin{matrix}\lambda_1 & & \\ & ... & \\ & & \lambda_n\end{matrix}\right]U^T\]
> 其中$U$是由拉普拉斯矩阵特征向量组成的矩阵，而$\lambda$代表着拉普拉斯矩阵的特征值

* 图领域的傅里叶变换
> 仿照传统领域下的傅里叶变换定义，我们就可以得到图领域的傅里叶变换
> \[F(\lambda_l)=\hat{f}(\lambda_l)=\sum^N_{i=1}f(i)u^*_l(i)\]
> $f$是图上的N维向量，$f(i)$表示节点i上对应的输入，$u_l(i)$表示第$l$个特征向量的第$i$个分量，特征值就对应了在不同基函数下对应的分量，也可以在一定程度上认为是对应的频率。$f$的图傅里叶变换就是与$\lambda_l$对应的特征向量$u_l$进行内积运算
> 进一步的，我们图傅里叶变换的矩阵形式写成：
> \[\left[ \begin{matrix} 
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
> \end{matrix}\right]\]
### 图卷积的快速计算
### 一阶近似与半监督学习
### deeper-insights
### 扩散卷积