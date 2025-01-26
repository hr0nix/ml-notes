# Applicability Limits of Rejection Sampling Fine-Tuning

Rejection Sampling Fine-Tuning (RFT) is an RL technique that gained popularity during the LLM revolution. The main reason for its popularity is that it's very simple to apply:
* Sample some trajectories from the model.
* Filter out those with low rewards.
* Fine-tune the model on the rest of the trajectories, reinforcing model's ability to produce high-reward trajectories.

This technique have been designed for RLHF bandit setting, where it works correctly.
However there have been some misguided attempts (including those made by me) to apply this technique to general stochastic multi-turn environments,
where it does not generally lead to policy improvement.

In this note I will show why this technique is correct when applied to bandits, and then provide an example of how it fails in a multi-step environment.

## Bandit case

For simplicity, let's consider the setup where the set of actions is finite and rewards are binary. We have a policy

$$\pi(a=a_i) = \pi_i$$

with

$$Q_{\pi}(a_i) = q_i,$$

the probability that $a_i$ will yield a reward of one. Applying RFT to this setup would mean sampling a lot of actions from $$\pi(a)$$ and leaving only those that resulted in a positive reward.
This process will result in a dataset with the number of occurences of each action proportional to $$\pi_i q_i$$. Assuming infinite training data and model capacity,
training on this dataset will yield the policy

$$\pi^*(a=a_i) = \frac{1}{Z} \pi_i q_i,$$

where

$$Z= \sum_i \pi_i q_i = V_{\pi},$$

the value of the initial policy. Question is, is $\pi^*(a)$ necessarily an improvement over $\pi(a)$? Turns out it is, and there is a beautiful way to show it using non-negativity of variance.
We need to show that

$$ V_{\pi^*} = \frac{1}{V_{\pi}} \sum_i \pi_i q_i^2 \geq V_{\pi}.$$

Let's define a random variable $X$ that takes the value $q_i$ with probability $\pi_i$. Then

$$\mathbb{E}[X] = \sum_i \pi_i q_i, \mathbb{E}[X^2] = \sum_i \pi_i q_i^2.$$

We know that

$$Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2 \geq 0,$$

therefore we have that

$$\sum_i \pi_i q_i^2 \geq (\sum_i \pi_i q_i)^2,$$

or, dividing both sides by $V_{\pi}$,

$$\frac{1}{V_{\pi}}\sum_i \pi_i q_i^2 \geq \sum_i \pi_i q_i,$$

which concludes the proof.

So, despite our bandit environment being stochastic, fine-tuning on positive outcomes is guaranteed not to hurt the initial policy and can even result in some gains!

## Multi-step case

But what about multi-step environments? First of all, if our environment is deterministic, and we additionally assume a finite horizon, we can just flatten the MDP into a bandit problem
by treating whole sequences of actions as bandit arms. Therefore, the theoretical argument from the previous section should hold in this case as well.

But what about multi-step stochastic environments? Intuition suggests that RFT should fail when environment stochasticity is present: when selecting for successful trajectories we will be
selecting both for cases where the policy did a good job, and where the good outcome was a result of sheer luck, and the latter can be bad strategies to imitate.
However we should be careful with intuitive arguments like this one: it can also be applied to the bandit case, where the rewards are stochastic, but it would be wrong there.
Let's instead try to build a simple multi-step counterexample where RFT would fail.

Let's consider a $2$-state MDP with states $S_0$ and $S_1$. Our policy can take action 
