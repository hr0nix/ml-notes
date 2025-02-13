# Some Interesting Properties of 1-step K-sample Lookahead Operator

1-step K-sample lookahead operator (which we will simply call lookahead operator from now on) in reinforcement learning is a sample-based policy improvement operator that works by first sampling $K$ action candidates from the base policy and then proceeding with the action with largest q-value:

$$L_{K, \pi}(a \mid s) = E_{a_1, \ldots, a_K \sim \pi(a \mid s)}\Bbb{1}\left[a = \arg \max_{a' \in \\{a_1, \ldots, a_K\\}} Q(s, a')\right].$$

It can be thought of as an imperfect approximation of the optimal max-q operator. It is very useful when dealing with large action spaces, where the exact maximum of the q-function over all actions cannot be computed. In this note, we will list various properties of this operator that might be useful when using it in practice.

### Lookahead operator is indeed a policy improvement operator

Follows from the policy improvement theorem:

$$E_{a \sim L_{K, \pi}(a \mid s)} Q(s, a) = E_{a_1, \ldots, a_K \sim \pi(a \mid s)} \max_{a' \in \\{a_1, \ldots, a_K\\}} Q(s, a') \geq E_{a \sim \pi(a \mid s)} Q(s, a) = V_{\pi}.$$


### Repeated application of lookahead increases the effective number of candidates ###

Suppose we define a new policy

$$\pi^{\ast}(a \mid s) = L_{K, \pi}(a \mid s).$$

What is then $L_{K, \pi^{\ast}}(a \mid s)$, i.e. the result of applying the lookahead operator twice? One way to think about it is this: one invocation of $\pi^{\ast}$ is effectively samping $K$ action candidates and returning the best of them. Lookahead operator $L_{K, \pi^{\ast}}(a \mid s)$ invokes $\pi^{\ast}$ $K$ times, so it effectively samples $K$ groups of $K$ candidates each, selects the best candidate in each group and then selects the best action across all $K$ groups. Since the same q-function is used to select both inside groups and across groups, this procedure is equivalent to simply doing lookahead with $K^2$ candidates.

Applying another lookahead on top of this policy will be equivalent to doing a single lookahead with $K^3$ candidates and so on. In other words, reaching the effective power of max-q improvement operator in an action space with $N$ actions requires just $O(log N)$ lookahead compositions, which is a very reasonable number even for large action spaces.


### Lookahead-based Reinforcement Learning

The above property suggests a straightforward RL algorithm:

1. Evaluate the current policy $\pi$ to get $q_{\pi}(s, a)$, e.g. by sampling some trajectories from $\pi$ and then training a model to predict return-to-go.
2. Distill $L_{K, \pi}(a \mid s)$ into a new policy $\pi^{\ast}$, e.g. by sampling some trajectories from $L$ and then applying supervised training on this data.
3. Set $\pi$ to $\pi^{\ast}$ and go to the first step unless done.

This algorithm
* Converges to the optimal policy in MDPs and does it fast even in large action spaces.
* Does not require to explicitly maximize the Q-function over actions, as this is covered by the repeated application of lookahead.
* Is very robust, as it only uses supervised learning as a learning subroutine. No sketchy non-stationary loss functions!

#### Repeated Policy Evaluation is Not Necessary

Perhaps the most surprising property of the algorithm is that instead of re-evaluating the policy 


