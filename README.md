# Fire

Preliminary implementation for the fire fighting problem with specified nodes to protect.

Let $ign$ be the ignition node, solve
$$
\min \qquad \sum_{i\in N}\sum_{k\in K}\sum_{r\in R} z_{i}^{kr}\tag{OF}
$$
Subject to
$$\sum_{ign,j\in A}x_{ign,j}=n-1$$
$$-\sum_{ij\in A}x_{ij}+\sum_{ji\in A}x_{ji}=1\quad \forall i\in N\setminus \{ign\}$$
$$x_{ij}\geq0\quad\forall ij\in A$$
$$t_{ign}=0$$
$$t_i\text{ free}\quad\forall i\in N\setminus \{ign\}$$
$$s_{ij}\geq0\quad\forall ij\in A$$
$$x_{ij}\leq (n-1)q_{ij}\quad\forall ij\in A$$
$$q_{ij}\in\{0,1\}\quad\forall ij\in A$$
$$\sum_{i\in N}\sum_{k\in K} z_{i}^{kr}\leq \sum_{k\in K}a_{kr}\quad\forall r\in R$$
