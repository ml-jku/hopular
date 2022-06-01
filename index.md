---
layout: default
title:  "Hopular: Modern Hopfield Networks for Tabular Data"
description: Blog post
date:   2021-10-23 23:00:00 +0200
usemathjax: true
---


$$
\newcommand{\Ba}{\boldsymbol{a}}
\newcommand{\Bp}{\boldsymbol{p}}
\newcommand{\Bu}{\boldsymbol{u}}
\newcommand{\Bv}{\boldsymbol{v}}
\newcommand{\Bx}{\boldsymbol{x}}
\newcommand{\By}{\boldsymbol{y}}
\newcommand{\Bz}{\boldsymbol{z}}
\newcommand{\Bw}{\boldsymbol{w}}
\newcommand{\Bg}{\boldsymbol{g}}

\newcommand{\BU}{\boldsymbol{U}}
\newcommand{\BV}{\boldsymbol{V}}
\newcommand{\BX}{\boldsymbol{X}}
\newcommand{\BY}{\boldsymbol{Y}}
\newcommand{\BZ}{\boldsymbol{Z}}
\newcommand{\BW}{\boldsymbol{W}}
\newcommand{\BS}{\boldsymbol{S}}
\newcommand{\BF}{\boldsymbol{F}}
\newcommand{\BG}{\boldsymbol{G}}
\newcommand{\BI}{\boldsymbol{I}}

\newcommand{\BXi}{\boldsymbol{\Xi}}
\newcommand{\Bxi}{\boldsymbol{\xi}}

\newcommand{\soft}{\mathrm{softmax}}

\newcommand{\rL}{\mathrm{L}}
$$


This post explains the paper "[Hopular: Modern Hopfield Networks for Tabular Data][arxiv-paper]".


**Hopular** ("Modern **Hop**field Networks for Tab**ular** Data") is a Deep Learning architecture for tabular data, where each layer is
equipped with continuous [modern Hopfield networks][ml-blog-hopfield].
Hopular is novel as it provides the original training set and the original input
at each of its layers. Therefore, Hopular refines the current prediction at every
layer by re-accessing the original training set like standard iterative learning algorithms.

{:refdef: style="text-align:center;"}
![not found](/assets/hopular_overview.svg){:width="100%"}
{:refdef}

A Hopular block stores two types of data:
- the whole training set
- the embedded input sample

The stored training set enables Hopular to find similarities across feature vectors and target vectors, while
the stored embedded input sample enables Hopular to determine dependencies between features and targets.

In real world, small-sized and medium-sized tabular datasets with less than 10,000 samples are ubiquitous.
Hitherto, Deep Learning underperformed on such datasets.
In contrast, Support Vector Machines (SVMs), Random Forests and, in particular, Gradient Boosting typically lead to higher performances than Deep Learning.
Gradient Boosting methods like XGBoost have the edge over other methods on most small-sized and medium-sized tabular datasets.

**Hopular surpasses Gradient Boosting, Random Forests, and SVMs but also state-of-the-art Deep Learning approaches to tabular data.**

## Table of Contents
1. [Motivation: Deep Learning Underperforms on Tabular Data](#motivation)
2. [Hopular: the new Deep Learning Architecture for Tabular Data](#architecture)
    1. [Input Layer: Embedding of the Input Sample](#embedding)
    2. [Hidden Layer: Hopular Block](#hblock)
    3. [Output Layer: Summarization of the Current Prediction](#summarization)
3. [Hopular Intuition: Mimicking Iterative Learning](#intuition)
    1. [Metric Learning for Kernel Regression by a Hopular Block](#kregression)
    2. [Linear Model with the AdaBoost Objective by a Hopular Block](#adaboost)
4. [Experiments](#experiments)
5. [Code and Paper](#codeAndPaper)
6. [Additional Material](#material)
7. [Correspondence](#correspondence)

## Motivation: Deep Learning Underperforms on Tabular Data <a name="motivation"></a>

In real world, small-sized and medium-sized tabular datasets with less than 10,000 samples are ubiquitous.
Their omnipresence can be witnessed at Kaggle challenges.
They are found in life sciences for:
- modeling certain diseases
- predicting bio-assay outcomes in drug design
- modeling environmental soil contamination

They are also found in most industrial applications for:
- predicting customer behavior
- controlling processes
- optimizing logistics
- recommending other products
- employing predictive maintenance

Deep Learning could not convince so far on small-sized and medium-sized tabular datasets. Therefore, we propose the **Hopular** Deep Learning architecture.

## Hopular: the new Deep Learning Architecture for Tabular Data <a name="architecture"></a>

The Hopular architecture consists 
of:
- input layer: embedding layer
- hidden layers: Hopular blocks
- output layer: summarization layer

Algorithm 1 shows the forward pass of Hopular for an original input sample $$\Bx$$.

{:refdef: style="text-align:center;"}
![not found](/assets/hopular_pseudocode.svg){:width="100%"}
{: refdef}

### Input Layer: Embedding of the Input Sample <a name="embedding"></a>

A categorical feature is encoded as a one-hot vector while a continuous feature is standardized.
The feature value, feature type, and feature position are all mapped to an $$e$$-dimensional embedding space.
All three embedding vectors are summed to a feature representation.
The input sample is represented by $$\By$$, which is the concatenation of all the input sample's feature representations. The current prediction $$\Bxi$$ is initialized by $$\By$$.

{:refdef: style="text-align:center;"}
![not found](/assets/hopular_embedding.svg){:width="100%"}
{: refdef}

The central component of the Hopular architecture is the Hopular block.

### Hidden Layer: Hopular Block <a name="hblock"></a>

A Hopular block consists of:
- Hopfield Module $$H_s$$ (sample-sample interactions)
- Hopfield Module $$H_f$$ (feature-feature interactions)
- Aggregation Block (result collection and information passthrough)

**(I) ---** Hopfield Module $$H_{s}$$. A continuous [modern Hopfield network][ml-blog-hopfield] for Deep Learning architectures is implemented
via the layer `HopfieldLayer` [Ramsauer et al., 2021][ramsauer:21-paper]; [Ramsauer et al., 2020][ramsauer:20-paper] 
with the training set as fixed stored patterns.
The current prediction $$\Bxi$$ serves as the input (the state vector) to Hopfield module $$H_{s}$$.
Thus, $$\Bxi$$ is interacting with the whole training data 
as described in Eq.$$~$$\eqref{eq:Hs}.
Therefore, the Hopfield module $$H_{s}$$ identifies sample-sample interactions
and can perform similarity searches like a nearest-neighbor search 
in the whole training data.

[//]: Include reference to iterative analogies.

The forward-pass for module $$H_{s}$$ with one Hopfield network and state $$\Bxi$$, 
learned weight matrices $$\BW_{\Bxi},\BW_{\BX}$$, $$\BW_{\BS}$$,
the stored training set $$\BX$$,
and a fixed scaling parameter $$\beta$$
is given as

$$\begin{align}\label{eq:Hs}\tag{1}
    H_s\left( \Bxi \right) \ &= \ \BW_{\BS} \  \BW_{\BX} \ \BX \ \soft \left( \beta \ \BX^{T} \ \BW_{\BX}^{T} \ \BW_{\Bxi} \ \Bxi \right).
\end{align}$$

The hyperparameter $$\beta$$ allows to steer
the nearest-neighbor-lookup of the
sample-sample Hopfield module $$H_{s}$$.
The module $$H_s$$ can comprise $$N$$ separate Hopfield networks $$H_{s}^{i}$$, where the module output is defined as

$$\begin{align}\label{eq:Hs_combined}\tag{2}
H_{s} \left(\Bxi \right) \ &= \ \BW_{G} \ \left( H_{s}^{1} \left( \Bxi \right)^{T}, \ldots,\ H_{s}^{N} \left( \Bxi \right)^{T} \right)^{T} \ ,
\end{align}$$

with vector $$\left( H_{s}^{1} \left( \Bxi \right)^{T}, \ldots,\ H_{s}^{N} \left( \Bxi \right)^{T} \right)^{T}$$
and a learnable weight matrix $$\BW_{G}$$.

**(II) ---** Hopfield Module $$H_{f}$$.
A continuous [modern Hopfield network][ml-blog-hopfield] for Deep Learning architectures is implemented 
via the layer `Hopfield` [Ramsauer et al., 2021][ramsauer:21-paper]; [Ramsauer et al., 2020][ramsauer:20-paper] 
with the embedded input features as stored patterns.
The current prediction $$\Bxi$$ serves as the input to Hopfield module $$H_{f}$$.
Prior to entering $$H_{f}$$, the current predition $$\Bxi$$ is reshaped
to the matrix $$\BXi$$ with the embedded input features as rows.
$$\BXi$$ interacts with the embedded features
of the original input sample
as described in Eq.$$~$$\eqref{eq:Hf}.
Therefore, the Hopfield module $$H_{f}$$ extracts and models
feature-feature and feature-target relations.
Thus, the current prediction $$\Bxi$$ interacts with the original input sample $$\By$$.

The forward-pass for module $$H_{f}$$ with one Hopfield network and state $$\BXi$$, 
learned weight matrices $$\BW_{\BXi},\BW_{\BY}$$, $$\BW_{\BF}$$,
the embedded input sample $$\BY$$,
and a fixed scaling parameter $$\beta$$
is given as

$$\begin{align}\label{eq:Hf}\tag{3}
    H_f \left(\BXi \right) \ &= \ \BW_{\BF} \ \BW_{\BY} \ \BY \ \soft \left( \beta \ \BY^{T} \ \BW_{\BY}^{T} \ \BW_{\BXi} \ \BXi \right).
\end{align}$$

$$H_{f}$$ may contain more than one 
continuous modern Hopfield network, which leads to an analog equation as Eq.$$~$$\eqref{eq:Hs_combined} of $$H_{s}$$.


**(III) ---** Aggregation Block.
The results of each Hopfield module are
combined via a residual connection with the current prediction $$\Bxi$$ (or its reshaped version $$\BXi$$ for $$H_{f}$$) and thereby refining it.
The current prediction $$\Bxi$$ is passed by the aggregation block to the next layer.

{:refdef: style="text-align:center;"}
![not found](/assets/hopular_block.svg){:width="100%"}
{: refdef}
 
The last layer of a Hopular architecture
is the output layer which maps the current prediction to the final prediction.

### Output Layer: Summarization of the Current Prediction <a name="summarization"></a>

Hopular is trained in a multi-task setting. Its objective is a weighted sum of two losses: 
- for predicting masked features of the input sample (BERT masking)
- for predicting the target of the input sample (standard supervised loss)

Thus, in addition to the target, also the masked features of the input sample must be predicted during training.
Therefore, the current prediction is a vector constructed by concatenating the current feature predictions and the current target prediction. 
The current prediction is mapped to the final prediction
by separately mapping each current feature prediction to the corresponding final prediction
as well as mapping the current target prediction to the final target prediction.

{:refdef: style="text-align:center;"}
![not found](/assets/hopular_summarization.svg){:width="100%"}
{: refdef}

## Hopular Intuition: Mimicking Iterative Learning <a name="intuition"></a>

A huge advantage of Hopular is that it can mimic iterative learning algorithms, in contrast to other Deep Learning methods for tabular data like NPTs and SAINT. Both NPTs and SAINT consider feature-feature and sample-sample interactions via their respective attention mechanisms which solely use the result of the previous layer.
In contrast, Hopular not only uses the result of the previous layer
but also the original input sample and the whole training set.

In every Hopular Block:
- the original input sample
- the whole training set

can be evaluated on the current prediction. This resembles computing the error for the input sample and updating the result on the whole training set.
<i>Sidenote: </i>if the original features are not overwritten, the current prediction can contain the original input.

### Metric Learning for Kernel Regression by a Hopular Block <a name="kregression"></a>

We consider the [Nadaraya-Watson kernel regression][nadaraya]. The training set is
$$\{(\Bz_1,\By_1),\ldots,(\Bz_N,\By_N)\}$$ 
with inputs $$\Bz_i$$ summarized by the input
matrix $$\BZ = (\Bz_1,\ldots,\Bz_N)$$ and labels $$\By_i$$ summarized
in the label matrix $$\BY=(\By_1,\ldots,\By_N)$$.
The kernel function is $$k(\Bz_i,\Bz)$$. The estimator $$\Bg$$ for $$\By$$ given $$\Bz$$ is:

$$
\begin{align}\tag{4}
\Bg(\Bz) \ &= \  \sum_{i=1}^N \By_i \ \frac{k(\Bz_i,\Bz)}{\sum_{i=1}^N  k(\Bz_i,\Bz)} \ .
\end{align}
$$

For vectors normalized to length $$1$$ and the exponential kernel $$k(\Bz_i,\Bz_j) = \exp(- \beta/2  \left\|\Bz_i - \Bz_j\right\| )$$, we have

$$
\begin{align}\tag{5}
  k(\Bz_i,\Bz_j) \ &= \ c \  \exp( \beta  \ \Bz_i^T \Bz_j )
\end{align}
$$

Therefore, the estimator is:

$$
\begin{align}\label{eq:estimator}\tag{6}
\Bg(\Bz) \ &= \   \BY \ \soft(\beta \ \BZ^T \Bz)
\end{align}
$$

Metric learning for kernel regression learns the kernel $$k$$ which is the distance function [Weinberger & Tesauro, 2007][weinberger-tesauro:07-paper].
A Hopular Block does the same in Eq.$$~$$\eqref{eq:Hs} via learning the weight matrices $$\BW_{\BX}$$ and $$\BW_{\Bxi}$$.
If we set in Eq.$$~$$\eqref{eq:estimator}:

$$
\begin{align}\tag{7}
\BZ^{T} = \BX^{T}\BW_{\BX}^{T}\ , \quad \Bz = \BW_{\Bxi}\ \Bxi\ , \quad \BY = \BW_{\BS} \BW_{\BX} \BX
\end{align}
$$

then we obtain Eq.$$~$$\eqref{eq:Hs}, with the fixed label matrix $$\BY$$.

### Linear Model with the AdaBoost Objective by a Hopular Block <a name="adaboost"></a>

The AdaBoost objective for a classification with a binary target $$y \in{} \{-1, +1\}$$ can be written as follows (Eq.$$~$$(3) and Eq.$$~$$(4) in [Shen & Li, 2010][shen-li:2010-paper]):

$$
\begin{align}\tag{8}
  \rL \ &= \  \ln \sum_{i=1}^{N}  \exp(- \ y_i \ g(\Bz_i) ) \ . 
\end{align}
$$

We use this objective for learning the linear model:

$$
\begin{align}\tag{9}
g(\Bz_i) \ &= \ \beta \ \Bxi^T \Bz_i \ . 
\end{align}
$$

The objective multiplied by $$\beta^{-1}$$ with $$\BY$$ as the diagonal matrix of targets $$\By_{i}$$ becomes

$$
\begin{align}\tag{10}
  \rL \ &= \  \beta^{-1} \ \ln \sum_{i=1}^{N}  \exp(-\beta \ y_i \ \Bxi^T \Bz_i ) \ 
  = \ \mathrm{lse}(\beta \ , -\BY \ \BZ^T  \Bxi)  \ , 
\end{align}
$$

where $$\mathrm{lse}$$ is the log-sum-exponential function.
The gradient of this objective is

$$
\begin{align}\tag{11}
  \frac{\partial \rL}{\partial \Bxi} \ &= \  
  - \ \BZ \ \BY \ \soft( - \ \beta \ \BY \ \BZ^T  \Bxi )  \ . 
\end{align}
$$

This is Eq.$$~$$\eqref{eq:Hs} with:

$$
\begin{align}\tag{12}
- \BY \BZ^{T} = \BX^{T}\BW_{\BX}^{T}\ , \quad \BW_{\Bxi} = \BI\ , \quad \BW_{\BS} = \BI
\end{align}
$$

Thus, a Hopular Block can implement a gradient descent update rule for a linear
classification model using the AdaBoost objective function.
The current prediction $$\Bxi$$ comes from previous layer.

## Experiments <a name="experiments"></a>



### Small-Sized Tabular Datasets

In this experiment we compare small-sized tabular datasets, where most of them have less than 500 samples.

**Methods Compared.**
We compare Hopular, XGBoost, CatBoost, LightGBM, NPTs, and other 24 machine learning methods
as described in [Wainberg et al., 2016][Wainberg:16-paper], [Klambauer et al., 2017][klambauer:17-paper].
The compared methods include 10 Deep Learning (DL) approaches.

**Datasets.**
Following [Klambauer et al., 2017][klambauer:17-paper], 
we consider UCI machine learning repository datasets with less than or equal to 1,000 samples as being <i>small</i>. 
We select a subset of 21 datasets, comprising 200 to 1,000 samples, from [Klambauer et al., 2017][klambauer:17-paper].
Of these, 13 datasets have 500 samples or less.

**Results.**
Across the considered UCI repository datasets
Hopular has the lowest median rank.
Therefore, <i>Hopular is the best performing method.</i>

{:refdef: style="text-align:center;"}
![not found](/assets/hopular_uci_results.svg){:width="100%"}
{: refdef}

### Medium-Sized Tabular Datasets

In this experiment we compare medium-sized tabular datasets of about 10,000 samples each.

**Methods Compared.**
We compare Hopular, NPTs, XGBoost, CatBoost, and LightGBM.

**Datasets.**
We select the datasets of [Schwartz-Ziv and Armon, 2021][ShwartzZiv:21-paper], 
where XGBoost performed better than Deep Learning methods that have been
designed for tabular data. We extend this selection by two datasets for regression: (a) <i>colleges</i> was already used for other Deep Learning methods for tabular data [Somepalli et al., 2021][somepalli:21-paper], and
(b) <i>sulfur</i> is publicly available and fits with its 10,082 instances well into
the existing collection of medium-sized datasets.

**Results.**
The next tables gives the accuracy for the different datasets and methods.
Hopular is the best performing method on 3 out of the 6 datasets.
The runner-up method, CatBoost, is twice the best method, whereas XGBoost once.
Over the 6 datasets, NPTs and XGBoost have a median rank of 4.5,
CatBoost and LightGBM of 2.5 and 2, respectively, and Hopular has a median rank of 1.5.
<i>On average over all 6 datasets, Hopular performs better than 
NPTs, XGBoost, CatBoost, and LightGBM.</i>

{:refdef: style="text-align:center;"}
![not found](/assets/hopular_medium_results.svg){:width="100%"}
{: refdef}

## Recap: Modern Hopfield networks

The associative memory of our choice are modern Hopfield networks for Deep Learning architectures 
because of their fast retrieval and high storage capacity 
as shown in [Hopfield networks is all you need](https://arxiv.org/abs/2008.02217).
The update mechanism of these modern Hopfield networks is equivalent to
the self-attention mechanism of Transformer networks.
However, modern Hopfield networks for Deep Learning architectures are more general
and have a broader functionality, of which
the Transformer self-attention is just one example. 
The according [Hopfield layers](https://github.com/ml-jku/hopfield-layers)
can be built in 
Deep Learning architectures for 
associating two sets, 
encoder-decoder attention,
multiple instance learning, or 
averaging and pooling operations. 
For details, see our blog [Hopfield Networks is All You Need](https://ml-jku.github.io/hopfield-layers/).

Modern Hopfield networks for Deep Learning architectures [Ramsauer et al., 2021][ramsauer:21-paper]; [Widrich et al., 2020][widrich:20-paper] are associative memories that have much higher storage capacity than classical Hopfield networks and can retrieve patterns with one update only.

## Code and Paper <a name="codeAndPaper"></a>

- [GitHub repository: hopular][github-repo]

- [Paper: Hopular: Modern Hopfield Networks for Tabular Data][arxiv-paper]

## Additional Material <a name="material"></a>

- [Paper: Hopfield Networks is All You Need][ramsauer:21-paper]

- [Blog: Hopfield Networks is All You Need][ml-blog-hopfield]

- [GitHub repository: hopfield-layers][github-hopfield]

- [Paper: Modern Hopfield Networks and Attention for Immune Repertoire Classification][widrich:20-paper]

- [Yannic Kilcher's video on modern Hopfield networks][kilcher-hopfield]

- [Blog post on Performers from a Hopfield point of view][ml-blog-performer]

- [Blog post on Energy-Based Perspective on Attention Mechanisms in Transformers][mcbal:20-blog]

For more information visit our homepage [https://ml-jku.github.io/][ml-blog].

## Correspondence <a name="correspondence"></a>

This blog post was written by Bernhard Schäfl and Lukas Gruber. 

Contributions by Angela Bitto-Nemling and Sepp Hochreiter.

Please contact us via schaefl[at]ml.jku.at


[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

[ml-blog]: https://ml-jku.github.io
[arxiv-paper]: https://arxiv.org/
[github-repo]: https://github.com/ml-jku/hopular

[ml-blog-hopfield]: https://ml-jku.github.io/hopfield-layers/
[ml-blog-performer]: https://iclr-blog-track.github.io/2022/03/25/Looking-at-the-Performer-from-a-Hopfield-point-of-view/
[github-hopfield]: https://github.com/ml-jku/hopfield-layers
[kilcher-hopfield]: https://www.youtube.com/watch?v=nv6oFDp6rNQ

[nadaraya]: https://en.wikipedia.org/wiki/Kernel_regression

[klambauer:17-paper]: https://arxiv.org/abs/1706.02515
[Kossen:21-paper]: https://arxiv.org/abs/2106.02584
[Wainberg:16-paper]: https://jmlr.org/papers/v17/15-374.html
[ShwartzZiv:21-paper]: https://openreview.net/forum?id=vdgtepS1pV
[somepalli:21-paper]: https://openreview.net/forum?id=nL2lDlsrZU

[aiweirdness:18-blog]: https://www.aiweirdness.com/do-neural-nets-dream-of-electric-18-03-02/
[belghazi:18-paper]: https://proceedings.mlr.press/v80/belghazi18a.html
[bommasani:21-paper]: https://arxiv.org/abs/2108.07258
[cheng:20-paper]: https://proceedings.mlr.press/v119/cheng20b.html
[dAmour:20-paper]: https://arxiv.org/abs/2011.03395
[geirhos:20-paper]: https://www.nature.com/articles/s42256-020-00257-z  
[ilharco:21-github]: https://github.com/mlfoundations/open_clip
[lapuschkin:19-paper]: https://www.nature.com/articles/s41467-019-08987-4 
[mcbal:20-blog]: https://mcbal.github.io/post/an-energy-based-perspective-on-attention-mechanisms-in-transformers
[poole:19-paper]: http://proceedings.mlr.press/v97/poole19a.html
[potter:12-paper]: https://www.frontiersin.org/articles/10.3389/fpsyg.2012.00113/full
[radford:21-paper]: http://proceedings.mlr.press/v139/radford21a.html
[ramsauer:20-paper]: https://arxiv.org/abs/2008.02217
[ramsauer:21-paper]: https://openreview.net/forum?id=tL89RnzIiCd  
[recht:19-paper]: http://proceedings.mlr.press/v97/recht19a.html
[sharma:18-paper]: https://aclanthology.org/P18-1238/
[taori:20-paper]: https://proceedings.neurips.cc/paper/2020/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html
[thomee:16-paper]: https://dl.acm.org/doi/10.1145/2812802
[wellman:93-paper]: https://ieeexplore.ieee.org/document/204911
[widrich:20-paper]: https://arxiv.org/abs/2007.13505
[weinberger-tesauro:07-paper]: https://proceedings.mlr.press/v2/weinberger07a
[shen-li:2010-paper]: https://ieeexplore.ieee.org/abstract/document/5432192
