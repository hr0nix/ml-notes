# Basic causality

In this note I will try to summarize my understanding of basics of causality. I will focus on viewing it through the prism of probabilistic modelling,
which, in my opinion, is one of the most powerful of efficient ways of thinking about ML problems.

## The setup

The classical ML setup is where we observe some pairs $(x, y)$ sampled from some unknown joint $P(x, y)$ and then,
given a new $x$, want to determine what is the corresponding $y$, usually in terms of $P(y \mid x)$. For instance, we might have some labelled images
and try to learn a labelling function that can work on previously unseed images.
We often do it by parametrizing our model with some $\theta$ and then maximizing the expected conditional log-likelihood,
where the expectation is taken over the joint:

$$`\theta^* = \arg \max_{\theta} \mathbb{E}_{(x, y) \sim P(x, y)} \log P(y \mid x, \theta)`.$$

Since $P(x, y)$ is unknown, in practice we usually estimate it via Monte-Carlo using our train set $D$, which is a sample from $P(x, y):

$$
`\theta^* \approx \arg \max_{\theta} \frac{1}{|D|} \sum_{(x, y) \in D} \log P(y \mid x, \theta).`
$$

Turns out, the success of this technique critically depends on where will the new $x$ come from:
* $x$ might be coming from the same marginal distribution $P(x) = \int_y P(x, y) dy$. This might be the case if we have some input stream of data,
label some random subset of it manually to learn the mapping, and then proceed to automatically label new items in the stream.
* $x$ might be coming from a different marginal $P^*(x)$[^1]. There might be several reasons for that:
  * Our training data is different from the data we intend to apply our model on. For instance, we might be training on ImageNet,
  but labelling images taken in the wild.
  * The act of making the prediction changes the input distribution through some feedback mechanism. A classical example would be behavioral cloning:
  we might observe some agent acting in the environment, learn what actions it takes in what states and then try to replicate that agent's behavior using
  our model. However, the states we will find ourselves in will depend on the actions we've taken and, depending on the quality of approximation
  and some other factors, might be very different from the states that the original agent faced.

The precise reasons for why the nature of $P^*(x)$ is important that can be expressed and quantified through the notion of _interventions_.

## Causal interventions

The notion of interventions is a way to formally define what does it mean to partially change the data distribution we are dealing with.
It starts from representing the original joint distribution as a product of conditionals. For example, let's consider a joint distribution $P(x, y, z, \theta)$,
where $\theta$ are model parameters, $x$ are the observed features, $y$ is the value of interest and $z$ are some unobserved features which can affect both $x$ and $y$
(such variables are called _hiddden confounders_ in causality literature).
This joint distribution can be written down as

$$`P(x, y, z, \theta) = P(\theta) P(z) P(x \mid z) P(y \mid x, z, \theta).`$$

An intervention is defined as changing some of the conditional distributions in this decomposition.
For instance, instead of using $x$ sampled from $P(x, y, z)$ we might decide to somehow produce our own, say, from $\psi(x)$.
Note that our $\psi(x)$ will not depend on $z$ because in practice we don't have access to $z$ when choosing $x$. Such change will result in a new joint

$$`P_{do(\psi(x))}(x, y, z, \theta) = P(\theta) P(z) \psi(x) P(y \mid x, z, \theta).`$$

We say that this joint is the result of _invervention_ $do(\psi(x))$ on $x$. Note that intervention can change the set of parents of $x$, or drop the parents completely,
as we did here. The most simple example of intervention is fixing $x$ to some constant value, in which case $\psi(x)$ is simply a Dirac delta.

Despite being quite simple, this notion allows us to analyze how model specified by $\theta^*$ will behave on new data.

## Analyzing different sources of $x$

If we want to apply our model on some $x$ sampled from $\psi(x)$, ideally the model should be optimal with respect
to the corresponding joint and conditional, i.e. we'd like to use

$$`\hat{\theta} = \arg \max_{\theta} \mathbb{E}_{(x, y) \sim P_{do(\psi(x))}(x, y)} \log P_{do(\psi(x))}(y \mid x, \theta),`$$

but that's not what we were solving before to obtain $\theta^*$. What is the relation between the two solutions?

### Case 1: marginal is the same

If our $x$ comes from $\psi(x)=P(x)$ as was the case in training data, then $P_{do(\psi(x))}(x, y)=P(x, y)$, $P_{do(\psi(x))}(y \mid x, \theta) = P(y \mid x, \theta)$
and, therefore, $\theta^* = \hat{\theta}$, so we get exactly the model we want.

### Case 2: no hidden variables, different marginal

Now let's assume that there are no hidden variables in our problem,
meaning that there is no unobserved variable $z$ such that either $x$ or $y$ depend on it. In this case out joint is

$$`P(x, y, \theta) = P(\theta) P(x) P(y \mid x, \theta).`$$

If we intervene on $x$ with $\psi(x)$, the resulting joint will be

$$`P_{do(\psi(x))}(x, y, \theta) = P(\theta) \psi(x) P(y \mid x, \theta),`$$

the marginal joint will be

$$`P_{do(\psi(x))}(x, y) = \int_{\theta} P(\theta) \psi(x) P(y \mid x, \theta) d\theta = \psi(x) P(y \mid x),`$$

and the conditional will be

$$`P_{do(\psi(x))}(y \mid x, \theta) = \frac{P_{do(\psi(x))}(x, y, \theta)}{P_{do(\psi(x))}(x, \theta)} =
\frac{P(\theta) \psi(x) P(y \mid x, \theta)}{P(\theta)\psi(x)} = P(y \mid x, \theta).`$$

Substituting these into the expression for $\hat{\theta}$ we can see that the desired model in this case is

$$`\hat{\theta} = \arg \max_{\theta} \mathbb{E}_{(x, y) \sim P_{do(\psi(x))}(x, y)} \log P(y \mid x, \theta).`$$

An important difference with $\theta^\*$ is what the expectation is taken over: ideally we want to optimize the expectation over $P_{do(\psi(x))}(x, y)$,
but in practice we are optimizing over $P(x, y)$. In the limit of infinite data and model capacity it should not matter: we will still be able to learn
$P(y \mid x, \theta)$ for every $x$. But in practice we might not have such luxury.
For instance, what can happen is that $P(x, y)$ frequently produces some $x$ which is rare under $\psi(x)$, and $\theta^\*$ will spend a large amount of its capacity
on such $x$ (maybe even memorize it!) at the expense of inputs we actually care about.

Luckily in this case there's a way to optimize the expectation of interest
using [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling)[^2]:

$$`\mathbb{E}_{(x, y) \sim P_{do(\psi(x))}(x, y)} \log P(y \mid x, \theta) = \int_{x, y} \log P(y \mid x, \theta) P_{do(\psi(x))}(x, y) dx dy =`$$

$$`= \int_{x, y} \log P(y \mid x, \theta) \frac{P_{do(\psi(x))}(x, y)}{P(x, y)} P(x, y) dx dy =`$$

$$`= \int_{x, y} \log P(y \mid x, \theta) \frac{\psi(x)}{P(x)} P(x, y) dx dy =`$$

$$`= \mathbb{E}_{(x, y) \sim P(x, y)} \log P(y \mid x, \theta) \frac{\psi(x)}{P(x)},`$$

which is something we can estimate in practice if we also learn $P(x)$. Note, however, that this estimate might have high variance if
$P(x)$ is small where $\psi(x)$ is large. This is only natural: we can't say much about $x$ for which we don't have a lot of training data.

### Hidden variables but no confounders

What if we now have some hidden variables that can affect either $x$ or $y$, but cannot affect both? In this case, the joint is

$$`P(x, y, \theta, z_x, x_y) = P(\theta) P(z_1) P(z_2) P(x \mid z_1) P(y \mid x, z_2, \theta).`$$

The joint after intervention is

$$`P_{do(\psi(x))}(x, y, \theta, z_x, x_y) = P(\theta) P(z_1) P(z_2) \psi(x) P(y \mid x, z_2, \theta).`$$

The corresponding marginal joint will be

$$`P_{do(\psi(x))}(x, y) = \int_{\theta, z_x, z_y} P_{do(\psi(x))}(x, y, \theta, z_x, x_y) d\theta dz_x dz_y = \psi(x) P(y \mid x),`$$

just as in case 2. The conditional $P_{do(\psi(x))}(y \mid x, \theta)$ will also be the same as in case 2.
So the presence of hidden variables which are not confounders is irrelevant for the problem formulation.

### A hidden confounder

Suppose now that there is an unobserved variable $z$ that affects both $x$ and $y$ simultaneously. In this case the joint is

$$`P(x, y, \theta, z) = P(\theta) P(z) P(x \mid z) P(y \mid x, z, \theta)`$$

and the joint after intervention is

$$`P_{do(\psi(x))}(x, y, \theta, z) = P(\theta) P(z) \psi(x) P(y \mid x, z, \theta).`$$

The conditional in this case is

$$`P_{do(\psi(x))}(y \mid x, \theta) = \frac{P_{do(\psi(x))}(x, y, \theta)}{P_{do(\psi(x))}(x, \theta)} = \frac{\int_z P(\theta) P(z) \psi(x) P(y \mid x, z, \theta) dz}{P(\theta)\psi(x)} =`$$

$$`= \frac{\int_z P(z) \psi(x) P(y \mid x, z, \theta) dz}{\psi(x)},`$$

so the 

[^1]: This is often referred to as _out-of-distribution_ (OOD) setting in ML literature.
[^2]: This technique is known as _propensity score matching_ in causality literature.
