# Causality and its implications for supervised learning

In this note I will try to explain when and why supervised learning fails because of causality-related effects. I will present it through the prism of probabilistic modelling, which I find to be a very efficient framework for analyzing ML problems.

## The setup

The classical supervised ML setup is where we observe some pairs $(x, y)$ sampled from an unknown joint distribution $P(x, y)$ and then,
given a new $x$, want to determine what is the corresponding $y$. For instance, we might have some labelled images
and try to learn a labelling function that can work on previously unseen images. A widely used approach is to describe the relationship between $x$ and $y$ in terms of $P(y \mid x, \theta^*)$, where $\theta^\*$ is a model found by maximizing the expected log-likelihood,
where the expectation is taken over the joint:

$$`\theta^* = \arg \max_{\theta} \mathbb{E}_{(x, y) \sim P(x, y)} \log P(y \mid x, \theta).`$$

Since $P(x, y)$ is unknown, in practice we estimate the expectation via Monte-Carlo using our train set $D$, which is a sample from $P(x, y)$:

$$
`\theta^* \approx \arg \max_{\theta} \frac{1}{|D|} \sum_{(x, y) \in D} \log P(y \mid x, \theta).`
$$

Turns out, the success of this technique critically depends on where will the new $x$ come from:
* $x$ might be coming from the same marginal distribution $P(x) = \int_y P(x, y) dy$. This might be the case if we have an i.i.d. stream of data,
label some random subset of it manually to learn the mapping, and then proceed to automatically label new items in the stream.
* $x$ might be coming from a different distribution $P^*(x)$[^1]. There might be several reasons for that:
  * Our training data is different from the data we intend to apply our model on. For instance, we might be training on ImageNet,
  but labelling images taken in the wild.
  * The act of making a prediction changes the input distribution through some feedback mechanism. One example is behavioral cloning where
  we observe an agent acting in some environment, learn what actions it takes in what states and then try to replicate the agent's behavior using
  our model. However, the states we will find ourselves in depend on the actions we've taken and, depending on the quality of approximation
  and some other factors, might be very different from the states that the original agent faced.

The precise reasons for why the nature of $P^*(x)$ is important can be expressed and quantified through the notion of _interventions_.

## Causal interventions

The notion of interventions is a way to formally define what does it mean to partially change the data distribution we are dealing with.
It starts from representing the original joint distribution as a product of conditionals. For example, let's consider a joint distribution $P(x, y, z, \theta)$,
where $\theta$ are model parameters, $x$ are the observed features, $y$ is the value of interest and $z$ are some unobserved features that can affect both $x$ and $y$
(such variables are called _hiddden confounders_ in causality literature).
This joint distribution can be written down as

$$`P(x, y, z, \theta) = P(\theta) P(z) P(x \mid z) P(y \mid x, z, \theta).`$$

This way of writing it down can be seen as specifying a causal process of generating our data: we first generate $\theta$ and $z$, then generate $x$ based on $z$, and, finally, generate $y$ based on previously generated values.

An intervention is defined as changing some of the conditional distributions in this decomposition to change the data generation process.
For instance, instead of using $x$ sampled from $P(x \mid z)$ we might decide to somehow produce our own, say, from $\psi(x)$, but keep the rest of the process intact. This change will result in a new joint

$$`P_{do(\psi(x))}(x, y, z, \theta) = P(\theta) P(z) \psi(x) P(y \mid x, z, \theta).`$$

We say that this joint is the result of _invervention_ $do(\psi(x))$ on $x$. Note that intervention can change the set of parents of $x$ (as long as we don't break causality), or drop the parents completely,
as we did here. The most simple example of intervention is fixing $x$ to some constant value, in which case $\psi(x)$ is simply a Dirac delta.

Despite being simple, this notion allows us to analyze how model specified by $\theta^*$ will behave on new data.

## Analyzing different problem setups

If we want to apply our model on some $x$ sampled from $\psi(x)$, ideally the model should be optimal with respect
to the corresponding joint and conditional, i.e. we'd like to use

$$`\hat{\theta} = \arg \max_{\theta} \mathbb{E}_{(x, y) \sim P_{do(\psi(x))}(x, y)} \log P_{do(\psi(x))}(y \mid x, \theta),`$$

but that's not what we were solving before to obtain $\theta^*$. What is the relation between the two solutions?

### Case 1: input distribution does not change

If our $x$ comes from $\psi(x)=P(x)$ as was the case in training data, then $P_{do(\psi(x))}(x, y)=P(x, y)$, $P_{do(\psi(x))}(y \mid x, \theta) = P(y \mid x, \theta)$
and, therefore, $\theta^* = \hat{\theta}$, so we get exactly the model we want.

### Case 2: different marginal, no hidden variables

For now let's assume that there are no hidden variables in our problem,
meaning that there is no unobserved variable $z$ such that either $x$ or $y$ depend on it. In this case out joint is

$$`P(x, y, \theta) = P(\theta) P(x) P(y \mid x, \theta).`$$

If we intervene on $x$ with $\psi(x)$, the resulting joint will be

$$`P_{do(\psi(x))}(x, y, \theta) = P(\theta) \psi(x) P(y \mid x, \theta),`$$

the marginal joint will be

$$`P_{do(\psi(x))}(x, y) = \psi(x) P(y \mid x),`$$

and the conditional will be

$$`P_{do(\psi(x))}(y \mid x, \theta) = \frac{P_{do(\psi(x))}(x, y, \theta)}{P_{do(\psi(x))}(x, \theta)} =
\frac{P(\theta) \psi(x) P(y \mid x, \theta)}{P(\theta)\psi(x)} = P(y \mid x, \theta).`$$

Substituting these into the expression for $\hat{\theta}$ we can see that the desired model in this case is

$$`\hat{\theta} = \arg \max_{\theta} \mathbb{E}_{(x, y) \sim P_{do(\psi(x))}(x, y)} \log P(y \mid x, \theta).`$$

An important difference with $\theta^\*$ is what the expectation is taken over: ideally we want to optimize the expectation over $P_{do(\psi(x))}(x, y)$,
but in practice we are optimizing it over $P(x, y)$. In the limit of infinite data and model capacity it should not matter: we will still be able to learn
$P(y \mid x, \theta)$ for every $x$. But in practice we might not have such luxury.
For instance, what can happen is that $P(x, y)$ frequently produces some $x$ which is rare under $\psi(x)$, and $\theta^\*$ will spend a large amount of its finite capacity
on such $x$ (maybe even memorize the answer for it!) at the expense of inputs we intent to apply the model on.

Luckily in this case there's a way to optimize the expectation of interest
using [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling)[^2]:

$$`\mathbb{E}_{(x, y) \sim P_{do(\psi(x))}(x, y)} \log P(y \mid x, \theta) = \int_{x, y} \log P(y \mid x, \theta) P_{do(\psi(x))}(x, y) dx dy =`$$

$$`= \int_{x, y} \log P(y \mid x, \theta) \frac{P_{do(\psi(x))}(x, y)}{P(x, y)} P(x, y) dx dy =`$$

$$`= \int_{x, y} \log P(y \mid x, \theta) \frac{\psi(x)}{P(x)} P(x, y) dx dy =`$$

$$`= \mathbb{E}_{(x, y) \sim P(x, y)} \log P(y \mid x, \theta) \frac{\psi(x)}{P(x)},`$$

which is something we can estimate in practice if we also learn $P(x)$. Note, however, that this estimate might have high variance if
$P(x)$ is small where $\psi(x)$ is large. This is only natural: we can't say much about $x$ for which we don't have a lot of training data.

### Case 3: hidden variables but no confounders

What if we now have some hidden variables that can affect either $x$ or $y$, but cannot affect both? In this case, the joint is

$$`P(x, y, \theta, z_x, x_y) = P(\theta) P(z_1) P(z_2) P(x \mid z_1) P(y \mid x, z_2, \theta).`$$

The joint after intervention is

$$`P_{do(\psi(x))}(x, y, \theta, z_x, x_y) = P(\theta) P(z_1) P(z_2) \psi(x) P(y \mid x, z_2, \theta).`$$

The corresponding marginal joint will be

$$`P_{do(\psi(x))}(x, y) = \int_{\theta, z_x, z_y} P_{do(\psi(x))}(x, y, \theta, z_x, x_y) d\theta dz_x dz_y = \psi(x) P(y \mid x),`$$

just as in case 2. The conditional $P_{do(\psi(x))}(y \mid x, \theta)$ will also be the same as in case 2.
So the presence of hidden variables which are not confounders is irrelevant for our analysis.

### Case 4: a hidden confounder

Suppose now that there is an unobserved variable $z$ that affects both $x$ and $y$ simultaneously. In this case the joint is

$$`P(x, y, \theta, z) = P(\theta) P(z) P(x \mid z) P(y \mid x, z, \theta)`$$

and the joint after intervention is

$$`P_{do(\psi(x))}(x, y, \theta, z) = P(\theta) P(z) \psi(x) P(y \mid x, z, \theta).`$$

The conditional in this case is

$$`P_{do(\psi(x))}(y \mid x, \theta) = \frac{P_{do(\psi(x))}(x, y, \theta)}{P_{do(\psi(x))}(x, \theta)} = \frac{\int_z P(\theta) P(z) \psi(x) P(y \mid x, z, \theta) dz}{P(\theta)\psi(x)} =`$$

$$`= \int_z P(z) P(y \mid x, z, \theta) dz.`$$

However it is not equal to the conditional induced by the joint before intervention, which is

$$`P(y \mid x, \theta) =  \frac{\int_z P(\theta) P(z) P(x \mid z) P(y \mid x, z, \theta) dz}{\int_z P(\theta) P(x \mid z) P(z) dz} = \frac{\int_z P(z) P(x \mid z) P(y \mid x, z, \theta) dz}{\int_z P(x \mid z) P(z) dz} = \int_z P(z \mid x) P(y \mid x, z, \theta) dz.`$$

So when there exists a hidden confounder $z$, we are not even optimizing the expectation of the right function! And, regretfully, there is no way around it: since $z$ is not observed, we cannot learn any conditionals involving it, so there is not way to learn $P_{do(\psi(x))}(y \mid x, \theta)$ directly.

But why are the two conditionals different? This is a consequence of the fact that there is a dependency between $x$ and $z$ in the joint before intervention, so the value of $x$ is informative about the likely values of $z$ there, changing in turn our beliefs about $y$. However, if we choose $x$ ourselves, its value does not carry information about $z$ and should not affect our beliefs about $y$. Note that the expressions become equal if $z$ is independent of $x$.

Here is an illustrative example of how this effect can arise:
* Suppose we want to learn an image classifier that can distinguish between images of cats and dogs.
* Our training data comes from two photographers, Alice and Bob. Alice mostly takes pictures of dogs while Bob prefers cats.
* There are two camera manufacturers, A and B. Camera made by each manufacturer takes images with a unique set of artifacts that can be noticed by a neural network. Alice mostly uses camera A, Bob prefers camera B.
* We didn't put the photograpers's identity in the dataset, so it's a hidden confounder.
* When we train a neural net on our dataset, it learns that if an image comes from camera A, it's probably taken by Alice, which means it's more likely to contain a dog.
* This is a perfectly valid conclusion for our data, but it doesn't hold if someone else uses camera A to take an image.
* Note that even if we ask Alice and Bob to take infinite amount of photos and train on them all, the problem does not go away. When in doubt, the model always associates photo taken by camera A with a higher probability of seeing a dog, because that's how we trained it to behave.

## Mitigating the hidden confounder problem

To reduce the discrepancy between what we are optimizing and what we should be optimizing, we need to make $P(y \mid x, \theta)$ and $P_{do(\psi(x))}(y \mid x, \theta)$ as similar as possible. I can think of several options for achieving that, all on the data collection side.

The most simple option but sometimes unavailable option is to avoid OOD setup at all, casting our problem to the case 1 from above. If your $x$ comes from the same distribution as the training data, presense of hidden confounders is irrelevant. One way to achieve that is to quickly deploy model of somewhat acceptable quality and start collecting and labelling data the model is being run on. This is not an option, however, in domains where labelling is expensive and data is scarce, like protein structure prediction.

We can also make the confounder observed, effectively casting our problem to the case 2 discussed above. It might not always be physically possible to obtain the value of $z$, but sometimes it is. In the example discussed above, we could have added photographer identity to features. Interestingly, in this problem setup we are implicitly assuming that everyone is either Alice or Bob, and the best thing we can do during inference is to integrate over both options.

Another option is to make $x$ as uninformative about $z$ as possible, essentially removing the confounding. For instance, we might want to ask Alice and Bob to use both cameras 50% of the time, this way image artefacts will no longer be informative about photographer's identity. Generally, comparing the conditionals with confounder before and after intervention tells us that the less informative $x$ is about $z$, the closer the two conditionals are, so we don't have to remove the confounding completely to see the performance benefits.

These are the only solutions I know of, if there are any other, I'd love to hear about them.

## Futher reading

For an interesting example of how the problem of having a hidden confounder can hurt generative models and behavior cloning, I recommend reading [Shaking the foundations: delusions in sequence models for interaction and control](https://arxiv.org/abs/2110.10819) by Ortega et. al. It also contains a more detailed introduction into causality.

[^1]: This is often referred to as _out-of-distribution_ (OOD) setting in ML literature.
[^2]: This technique is known as _propensity score matching_ in causality literature.
