As many of us know, uncertainty can be decomposed into aleatoric and epistemic uncertainty, where
* Aleatoric uncertainty corresponds to the irreducible stochasticity, such as the results of a coin toss.
* Epistemic uncertainty represents the uncertainty that can be reduced to zero if we had infinite data about the problem.

But what does it mean precisely? Turns out this decomposition can be derived by analyzing the entropy of the predictive posterior.
Imagine we have a dataset of data points $X$ and we want to to infer a distribution over a new datapoint $x$ using a family of models,
which members will be denoted by $\theta$. The predictive posterior for $x$ is

$$
P(x \mid X=X) = \int_{\theta} P(x \mid \theta) P(\theta \mid X=X) d\theta
$$

Its entropy $H[x \mid X = X]$ represents the total uncertainty about the value of $x$ given the data $X$ that we have.

To decompose this uncertainty we need two properties of entropy:
* $H[A, B]$ = H[A] + H[B] - I[A, B]$, which provides a decomposition of the entropy of the joint in terms of entropies of the marginals and mutual information about the variables.
* $H[A \mid B] = H[A, B] - H[B]$, meaning that conditional entropy is a difference between our uncertainty about the joint distribution and the marginal distribution of the condition.

Applying the first property to the predictive posterior, we get
$$
H[x, \theta \mid X=X] = H[x \mid X=X] + H[\theta \mid X=X] - I[x, \theta \mid X=X]
$$, or
$$
H[x \mid X=X] = H[x, \theta \mid X=X] - H[\theta \mid X=X] + I[x, \theta \mid X=X]
$$, which using the second property can be transformed into
$$
H[x \mid X=X] = H[x \mid \theta, X=X] + I[x, \theta \mid X=X]
$$.

Let's analyze the terms we've got here. 
$$
H[x \mid \theta, X=X] = \int_{\theta} H[x \mid \theta=\theta, X=X] P(\theta \mid X=X) d\theta,
$$
so here, given a model $\theta$ we compute the uncertainty about $x$ from the point of view of the model, and average it over all models, taking into account how likely they are.
This is *aleatoric* uncertainty.

$$
I[x, \theta \mid X=X] = H[\theta \mid X=X] - H[\theta \mid x, X=X],
$$
where we decomposed mutual information into a difference of entropies. Note that this term is zero when adding $x$ to $X$ does not provide any new information about $\theta$,
meaning that we've learned all we could from $X$. This is *epistemic* uncertainty.

