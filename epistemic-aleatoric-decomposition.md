As you probably know, uncertainty about a value can be decomposed into aleatoric and epistemic uncertainty, where
* Aleatoric uncertainty corresponds to the irreducible stochasticity in the value, such as its dependecy on results of a coin toss.
* Epistemic uncertainty represents the uncertainty that can be reduced to zero if we have infinite data about our problem.

But what does it mean in terms of math? Turns out this decomposition can be derived by analyzing the entropy of the predictive posterior.
Imagine we have a dataset of data points $X$ and we want to to infer a distribution over a new datapoint $x$ using a family of models,
which members will be denoted by $\theta$. The predictive posterior for $x$ is

$$
P(x \mid X) = \int_{\theta} P(x \mid \theta) P(\theta \mid X) d\theta
$$

Its entropy $H[x \mid X = X]$ represents the total uncertainty about the value of $x$ given the data $X$ that we have.

To decompose this uncertainty we will use two properties of entropy:
* $H[A, B] = H[A] + H[B] - I[A, B]$, which provides a decomposition of the entropy of the joint in terms of entropies of the marginals and the mutual information between the variables of interest.
* $H[A \mid B] = H[A, B] - H[B]$, meaning that conditional entropy is a difference between our uncertainty about the joint and marginal values.

Both properties can be easily derived from the definition of entropy.

Applying the first property to the joint posterior, we get
$$H[x, \theta \mid X=X] = H[x \mid X=X] + H[\theta \mid X=X] - I[x, \theta \mid X=X].$$
We can rearrange the terms to express the entropy of the predictive posterior:
$$H[x \mid X=X] = H[x, \theta \mid X=X] - H[\theta \mid X=X] + I[x, \theta \mid X=X],$$
which, using the second property, can be transformed into
$$H[x \mid X=X] = H[x \mid \theta, X=X] + I[x, \theta \mid X=X].$$

Let's analyze the terms we've got here. By definition,
$$H[x \mid \theta, X=X] = \int_{\theta} H[x \mid \theta=\theta, X=X] P(\theta \mid X) d\theta,$$
so here, given a model $\theta$ we compute the uncertainty about the value of $x$ from the point of view of that model, and average this uncertainty over all models, taking into account how likely they are given the data.
This is *aleatoric* uncertainty.

$$I[x, \theta \mid X=X] = H[\theta \mid X=X] - H[\theta \mid x, X=X],$$
where we decomposed mutual information into a difference of entropies. Note that this term is zero only when adding $x$ to $X$ does not provide any new information about $\theta$,
meaning that if we've learned all we could from $X$, it vanishes. This is *epistemic* uncertainty.

