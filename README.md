# Fire

Cut

Let $ign$ be the ignition node, $p\in P$ be the set of nodes to be protected. Wirte $SPT_p$ as the shortest path from $ign$ to $p$ and $c_{SPT_p}$ be the corresponding travel cost.
It suffices
$$\sum_{i\in SPT_p}\sum_{k|b_k<C_{SPT_p}}z_i^k\geq 1$$


***

Preliminary implementation for the fire fighting problem with specified nodes to protect.

Let $ign$ be the ignition node, solve
$$\min \qquad \sum_{i\in N}\sum_{k\in K}\sum_{r\in R} z_{i}^{kr}\tag{OF}$$
Subject to
$$\sum_{ign,j\in A}x_{ign,j}=n-1$$
$$-\sum_{ij\in A}x_{ij}+\sum_{ji\in A}x_{ji}=1\quad \forall i\in N\setminus \{ign\}$$
$$x_{ij}\geq0\quad\forall ij\in A$$
$$t_{ign}=0$$
$$t_i\text{ free}\quad\forall i\in N\setminus \{ign\}$$
$$s_{ij}\geq0\quad\forall ij\in A$$
$$x_{ij}\leq (n-1)q_{ij}\quad\forall ij\in A$$
$$q_{ij}\in\{0,1\}\quad\forall ij\in A$$
$$\sum_{i\in N}\sum_{k\in K} z_{i}^{kr}\leq \sum_{k\in K}a_{kr}\quad\forall r\in R \tag{Res1}$$
$$\sum_{r\in R}\sum_{k\in K} z_{i}^{kr}\leq 1\quad\forall i\in N$$
$$\sum_{i\in N} z_{i}^{1r}+o_{1r}=a_{1r}\quad\forall r\in R \tag{Res2}$$
$$\sum_{i\in N}\sum_{r\in R} z_{i}^{kr}+o_{kr}=a_{kr}+o_{k-1,r}\quad k=2,\cdots,|K|\quad\forall r\in R \tag{Res3}$$
$$z_{i}^{kr}\leq 1+\frac{t_i-b_k}{b_k}\quad\forall i\in N,k\in K, r\in R$$
$$o_{kr}\geq0\quad\forall k\in K, r\in R \tag{Res4}$$
$$z_{i}^{kr}\in\{0,1\}\quad\forall i\in N,k\in K, r\in R$$
$$t_j-t_i+s_{ij}=c_{ij}+\sum_{r\in R}\left(\Delta_r\cdot\sum_{k\in K}z_{i}^{kr}\right)\quad\forall ij\in A\tag{Res5}$$
$$s_{ij}\leq \left((n-1)\cdot c_{\max}+(|R|-1)\cdot \left(\max_{r\in R}\Delta_r\right)\right)(1-q_{ij})\quad\forall ij\in A\tag{Res6}$$


Following changes are made:
  1. Constraints marked with (Res) are now modified resource constraints, where (Res2), (Res3) and (Res4) naturally follows by adding an extra dimension to account for the index.
  2. (Res1) now limits the number of resources allocated must not exceed the total amount given, for each type of resources
  3. (Res5) follows the same constraint but also accounts for resource-specific dealy in fire spreading. So the slackness update is bounded by maximum delay as modified in (Res6)

See the [original model](test.ipynb) and the [modified model](model_protect.py). 

### TODO
- [ ] Finish [WIP](model.ipynb)
