# Some Interesting Properties of 1-step K-sample Lookahead Operator

1-step K-sample lookahead operator (which we will simply call lookahead operator from now on) in reinforcement learning is a sample-based policy improvement operator that works by first sampling $K$ action candidates from the base policy and then proceeding with the action with largest q-value:

$$L_{K, \pi}(a \mid s) = E_{a_1, \ldots, a_K \sim \pi(a \mid s)}\Bbb{1}\left[a = \arg \max_{a' \in \\{a_1, \ldots, a_K\\}} Q(s, a')\right].$$

It can be thought of as an imperfect approximation of the optimal max-q operator. It is very useful when dealing with large action spaces, where the exact maximum of the q-function over all actions cannot be computed. In this note, we will list various properties of this operator that might be useful when using it in practice.

### Lookahead operator is indeed a policy improvement operator

This is trivial to show using the policy improvement theorem:

$$E_{a \sim L_{K, \pi}(a \mid s)} Q(s, a} = E_{a_1, \ldots, a_K \sim \pi(a \mid s)}\Bbb{1}\left[a = \arg \max_{a' \in \\{a_1, \ldots, a_K\\}} Q(s, a')\right] Q(s, a)$$
