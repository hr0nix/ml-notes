# The Applicability Limits of Rejection Sampling Fine-Tuning

Rejection Sampling Fine-Tuning (RFT) is an RL technique that gained popularity during the LLM revolution. The main reason for its popularity is that it is very simple to apply:
* Collect some trajectories from the model.
* Filter out those with low rewards.
* Fine-tune the model on the rest of the trajectories, reinforcing model's ability to produce high-reward trajectories.

In this note we will discuss under which conitions this technique actually guarantees policy improvement, and when it can result in a weaker policy compared to the data collection policy.

## A quick side note

Originally I intended to write this note to prove that RFT is _not_ a policy improvement in multi-step stochastic environments, as my intuition strongly suggested that learning solely from successful trajectories would imply maximising over environment stochasticity, which will result in mistakenly attributing success to actions where it was due to environment. The reality however turned out to be slightly more complex:
* For a certain reward structure (sparse binary rewards) this method results in a guaranteed policy improvement.
* However for general reward structures (arbitrary rewards on any step) this method does not yield improvements even for bandits, so multi-stepness does not matter.
Yet another reminder not to rely on intuition too much.

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

the value of the initial policy.

Is $\pi^*(a)$ necessarily an improvement over $\pi(a)$? Turns out it is, and there is a beautiful way to show it using non-negativity of variance.
We need to show that

$$ V_{\pi^\ast} = \frac{1}{V_{\pi}} \sum_i \pi_i q_i^2 \geq V_{\pi}.$$

Let's define a random variable $X$ that takes the value $q_i$ with probability $\pi_i$. Then

$$E[X] = \sum_i \pi_i q_i,$$

$$E[X^2] = \sum_i \pi_i q_i^2.$$

We know that

$$Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2 \geq 0,$$

therefore we have that

$$\sum_i \pi_i q_i^2 \geq (\sum_i \pi_i q_i)^2,$$

or, dividing both sides by $V_{\pi}$,

$$\frac{1}{V_{\pi}}\sum_i \pi_i q_i^2 \geq \sum_i \pi_i q_i = V_{\pi},$$

which concludes the proof.

So, despite our bandit environment being stochastic, fine-tuning on positive outcomes is guaranteed not to hurt the initial policy and can even result in some gains!

## Multi-step case

Let us now extend our MDP with finite actions and binary rewards to the multi-step case. We will consider the case of sparse rewards received when upon the termination of an episode.

First, let's figure out what our RFT policy will look like in multi-step case, e.g. how many times each $(s, a)$ pair will occur in the RFT dataset. Whenever we get into state $s$ during data collection, we will act with action $a_i$ with probability $\pi_i$, and then eventually get non-zero reward in $Q_{\pi}(s, a)$ fraction of all cases where $a_i$ was chosen in state $s$. Therefore, just as in the bandit case,

$$\pi^*(a=a_i \mid s) = \frac{1}{V_{\pi}(s)} \pi(a=a_i \mid s) Q_{\pi}(s, a_i).$$

According to the policy improvement theorem, to show that $\pi^\ast$ is an improvement over $\pi$, we need to show that

$$E_{a \sim \pi^\ast(a \mid s)}[Q_{\pi}(s, a)] = \sum_i \pi(a=a_i \mid s) Q^2_{\pi}(s, a_i) \geq V_{\pi}(s),$$

which is what we've already established in the bandit case.

## General setting

One can consider a more general setting, where rewards can be arbitrary, and we train on trajectories with rewards exceeding some threshold $T$. In this setting policy improvement no longer holds, and it's quite easy to build a counterexample even for the bandit case.

Let's consider a bandit with two actions, $a_1$ and $a_2$. Let $a_1$ always result in $0.1$ reward, so $Q(a_1)=0.1$. Let $a_2$ yield $-1$ reward in $\frac{1}{2}$ of all cases, and $+1$ otherwise. Therefore $Q(a_2)=0$.

Let's now consider a data collection policy $\pi$ that chooses either $a_1$ or $a_2$ with probability $\frac{1}{2}$. The value of this policy is

$$V_{\pi} = \frac{1}{2} \times 0.1 + \frac{1}{2} \times 0 = 0.05.$$

It's easy to see that if we use $T > 0.1$ in RFT, we will learn a deterministic policy $\pi^\ast$ that always chooses $a_2$ (as it's the only action yielding large enough reward) with $V_{\pi^\ast} = 0$, which is worse. The reason for failure is that we essentially attribute high reward of $1$ to the choice of action $a_2$, maximising over the stochasticity of the environment instead of averaging over it.

Therefore, RFT with arbitrary reward structures should be used with great caution, as it can result in a policy that performs worse than the baseline.
