# Paper Notes

## Planning

### RL Related

> ***Rethinking Closed-loop Training for Autonomous Driving*** **ECCV'22** [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990259.pdf)

**Key Insight:** There is a lack of understanding of how to build effective training benchmark for closed-loop training (What type of scenarios do we need to learn to drive safely). Popular RL algs cannot achieve satisfactory performance in the context of AD, as they lack long-term planning and take an extremely long time to train. So it proposes **TRVAL**, an RL-based driving agent that performs planning with **multistep look-ahead** and **exploits cheaply generated imagined data** for efficient learning.

**Method:** Direct output **control signal may lack long-term reasoning**. Explicit model rollout can be prohibitively expensive.

Combining aspects of model-fre and model-based approaches: 1. reason into the future by directly costing trajectories without explicit rollouts 2. learn from imagined (i.e. counterfactual) experiences from an approximate world model.

Define action as a trajectory, which navigates the ego-vehicle for the next $T$ time steps. Decompose the the cost of following a trajectory into a short-term *cost-to-come*, $R_\theta(s,\tau)$, defined over the next $T$ timesteps, and a long-term *cost-to-go* $V_\theta(s,\tau)$.

Using imagined counterfactual rollouts (non-reactive) as supervision only for short-term cost-to-come.

**Experiment:** Unknown highway dataset. More variations, target scenarios are important for closed-loop agent training.



> ***UMBRELLA: Uncertainty-Aware Model-Based Ofﬂine Reinforcement Learning Leveraging Planning*** **NeurIPS'21 Workshop (Best Paper)** [Paper](https://arxiv.org/abs/2111.11097)

**Key insight:** Extention of MBOP, which allows for a simple extension of the reward function and the incorporation of state constraints. Besides, planning with the learned dynamics model enhances **interpretability**. However, MBOP uses a simple deterministic dynamics model, which ignores aleatoric uncertainty. Besides, it operates in a fully observable setting.

**Method:** It uses **n-th order history** to represent the state, transforming the POMDP into an MDP. To model different futures, it learns stochastic forward dynammics models $f_{m,\theta}:\mathcal{S}\times\mathcal{A}\times\mathcal{Z}\rightarrow\mathcal{S}\times\mathbb{R}$, which is a CVAE. The latent variable $\mathbf{z}_t$ models different future predictions (other agents' behavior variations) and makes sure the output is non-deterministic to the input. 

This works follows the nth-order history approach to account for states, which are not fully observable and merely estimated from the last observations $\mathbf{o}_{t-n_c:t}$.

The behavior cloning model takes the current state and the $n_c$ previous actions (the output can be more smooth) as input and output the action. Also proposes Umbrella-P.

**Experiments:** NGSVM and CARLA. 

Limitation of model-based offline planning: The reward function does not exactly represent the human driving style; the performance is limited by the unimodal BC policy (multi-modal BC policies).



> ***Offline Reinforcement Learning for Autonomous Driving with Safety and Exploration Enhancement*** **NeurIPS'21 Workshop** [Paper](https://arxiv.org/abs/2110.07067)

Simple extension of batch-constrained Q-learning: Injecting parameter noises and lyapunov stability.



> ***Motion Planning for Autonomous Vehicles in the Presence of Uncertainty Using Reinforcement Learning*** **IROS'21** [Paper](https://arxiv.org/abs/2110.00640)

**Key insight:** Previous methods end up in conservative planning and expensive computation. This paper proposes a RL based solution to manage uncertainty by **optimizing for the worst case outcome (using quantile regression).** It is built on top of the distributional RL with its policy optimization maximizing the stochastic outcomes' lower bound.



> ***Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning*** **arXiv'20** [Paper](https://arxiv.org/abs/2001.08726) | [Code](https://github.com/cjy1992/interp-e2e-driving)

**Key insight:** A sequential latent environment model is introduced and learned jointly with the reinforcement learning process. With this latent model, a semantic birdeye mask can be generated, which is enforced to connect with a certain intermediate property for the purpose of explaining the behaviors of learned policy. 

**Method:** Variational inference. Reconstruct BEV mask (only get sensor inputs) and sensor inputs. MaxEnt RL can be interpreted as learning as learning a PGM using optimal variable with soft probability $p(O_t=1|z_t,a_t)=\exp(r(z_t,a_t)$.

**Experiments:** Carla.



> ***Model-free Deep Reinforcement Learning for Urban Autonomous Driving*** **arXiv'19** [Paper](https://arxiv.org/abs/1904.09503) | [Code](https://github.com/cjy1992/interp-e2e-driving)

**Key insight:** Current RL methods generally do not work well on complex urban scenarios. This paper proposes a framework to enable model-free deep RL in challenging urban autonomous driving scenarios. It designs a specific input representation and use visual encoding to capture the low-dimensional latent states (BEV & VAE). Several tricks: modified exploration strategies, frame skip, network architectures and reward designs.

**Experiments:** Carla.



> ***Learning to Drive in a Day*** **arXiv'18** [Paper](https://arxiv.org/abs/1807.00412)



### IL Related

> ***Guided Conditional Diffsuion for Controllable Traffic Simulation*** **arXiv'22** [Paper](https://arxiv.org/abs/2210.17366)

**Key insight:** The control-realism tradeoff has long been a problem in AD simulators. This paper proposes to use diffusion model (similar to that in Diffuser) guided by signal temporal logic (STL) measure to manage the tradeoff. It further extends the denoising step (**batched denoising**) to the multi-agent setting and enable interaction-based rules like collision avoidance.

**Method:** It models the trajectory, but only the action trajectory, since the state trajectory is derived by the dynamics model (therefore bypass the start state inpainting problem). Then it uses the quantitative STL measure to guide the diffusion model during sampling to create realistic controllable traffic scenes.



> ***Model-Based Imitation Learning for Urban Driving*** **NeurIPS'22** [Paper](https://arxiv.org/abs/2210.07729) | [Code](https://github.com/wayveai/mile)

**Key insight:** The paper presents MILE: a model-based imitation learning approach to **jointly learn a model of the world and a policy** for autonomous driving. The model can predict diverse and plausible states and actions, that can be interpretably decoded to BEV semantic segmentation. It can also execute complex driving manoeuvres from plans entirely predicted in imagination. Do **not assume access to GT physical states (position, velocity) or to an offline HD map** for scene context. MILE is the first camera-only method that models static, dynamic scenes and ego-behavior in an urban driving environment.

**Method:** Variational inference. The goal is to infer the latent dynamics $(\mathbf h_{1:T},\mathbf s_{1:T})$ that generate the observations, the expert actions, and the BEV labels. The latent dynamics contains a deterministic history $\mathbf h_t$ and a stochastic state $\mathbf s_t$.

Deteministic dynamics: $\mathbf h_t=f_\theta(\mathbf h_t,\mathbf s_t)$. The prior distribution of the stochastic state $p(\mathbf s_t|\mathbf h_{t-1},\mathbf s_{t-1})$. The posterior distribution $q(\mathbf s_t|\mathbf o_{\leq t},\mathbf a_{<t})$. Generative models takes $\mathbf h_{t-1},\mathbf s_{t-1}$ as inputs. The observation is encoded by an encoder $\mathbf x_t'=e_\phi(\mathbf o_t)$ which lifts image features to 3D, pools them to BEV, and compress to 1D vector. Finally, the observation embedding is the concatenation of the image feature, route map feature (goal) and speed feature.

**Experiment:** Carla challenge.



> ***Hierarchical Model-Based Imitation Learning for Planning in Autonomous Driving*** **arXiv'22** [Paper](https://arxiv.org/abs/2210.09539)

**Key insight:** It demonstrates the first large-scale application of **model-based generative adversarial imitation learning (MGAIL)** to the task of dense urban self-driving. It augments the MGAIL using a hierarchical model to enable generalization to arbitrary goal routes, and measure performance using a closed-loop evaluation framework with simulated interactive agents.

**Method:** A common challenge with common IL is **covariate shift,** also known as "DAgger problem". The fundamental problem is that its open-loop training will incur compounding error at each time step.  Therefore, the paper proposes to use MGAIL (model-based generative adversarial imitation learning) to conduct closed-loop training. Typically, it uses the deltas action model to enable differentiable policy update. 

Another problem is that the planning problem should be **goal-directed.** So the paper proposes an hierarchical structure for planning: the high-level module uses bidirectional A* to produce a goal route; the low-level module uses Transformer and cross attention conditioned on the **goal route,** scene context, **traffic light**  to output the discriminator score for GAIL as well as the action in the next timestep.

**Experiment:** Waymo's own dataset from a fleet of their vehicles operating in San Francisco. Propose to measure the average performance as well as the performance on difficult and uncommon situations (i.e. the long-tail performance).



> ***ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning*** **ECCV'22** [Paper](https://arxiv.org/abs/2207.07601) | [Code](https://github.com/OpenPerceptionX/ST-P3)

**Key insight:** To better predict the control signals and enhance user safety, this paper proposes an end-to-end approach that benefits from joint spatial-temporal feature learning. 

**Method:** After getting the context feature from the perceiving stage, ST-P3 adopts dual pathway probabilistic future modeling to get the context feature at future timesteps. Specifically, one branch models the future uncertainty as diagonal Gaussians, the other branch takes account into the historical features. Based on the future context features, a cost volume head is desiged to output the heatmap at future timesteps. Finally, it uses three costs to compose the total cost: safety cost for colliding with environment agents, a learned cost through the output heatmap, a trajectory cost for its comfort and smoothness. It uses sampler to output the final trajectory. Specifically, for the planning stage it provides the **high-level command (goal position) and the front-view traffic lights** into a GRU to refines the planned trajectory.

**Experiment:** Open-loop nuScenes and closed-loop Carla.



> ***PlanT: Explainable Planning Transformers via Object-Level Representations*** **CoRL'22** [Paper](https://arxiv.org/abs/2210.14222) | [Code](https://github.com/autonomousvision/plant)

**Key insight:** Existing learning-based planners typically extract features from dense, high-dimensional grid representations containing all vehicel and road context information. This paper proposes PlanT based on imitation learning with a compact object-level input representation. The experiment results indicate that PlanT can focus on the most relevant object in the scene, even when the object is geometrically distant.

**Method:** The most important part is the **tokenization** for the transformer input. Typically, it represents the  scene using a set of objects, with vehicles and segments of the route each being assigned an oriented bounding box in BEV space. The vehicle set just contains all vehicles in the BEV, while the route segments boxes are got by sampled a dense set along the route the route ahead of the ego vehicle. For each object, it has 6 attributes associated with the relative position, length/width of the bounding box, the orientation, and the speed for the vehicle or the order for the route segments.

It introduces a **[CLS] token to attentively sum from all the other agents.** The input transformer then projects all the tokens. It uses the embedding of [CLS] into a GRU to generate multi-step future waypoints auto regressively. Besides, it also uses the embedding of the environment agents to regress their future bounding boxes accordingly as an **auxiliary supervision**.

**Experiment:** Carla *Longest6*.



> ***End-to-End Urban Driving by Imitating a Reinforcement Learning Coach*** **CVPR'21** [Paper](https://arxiv.org/abs/2108.08265) | [Code](https://github.com/zhejz/carla-roach)

**Key insight:** On-policy demonstrations from humans is non-trivial. Labeling the targets given sensor measurements turns out to be a challenging task for humans. Only sparse events like human interventions are recorded, which is better suited for RL than IL methods. This paper focues on automated experts, in which the IL problem can be seen as a knowledge transfer problem. The paper proposes Roach, an RL expert that maps BEV images to continuous actions, which can provide **action distributions, value estimations and latent features as supervision.**

**Method:** RL expert: PPO + entropy and exploration loss. IL agent: DA-RB (CILRS + DAGGER). Image and measurement inputs.

**Experiment:** CARLA leaderboard.



> ***Perceive, Predict, and Plan: Safe Motion Planning Through Interpretable Semantic Representations*** **ECCV'20** [Paper](https://arxiv.org/abs/2008.05930)



> ***DSDNet: Deep Structured self-Driving Network*** **ECCV'20** [Paper](https://arxiv.org/abs/2008.06041)



> ***Jointly Learnable Behavior and Trajectory Planning for Self-Driving Vehicles*** **IROS'19** [Paper](https://arxiv.org/abs/1910.04586)



> ***End-to-end Interpretable Neural Motion Planner*** **CVPR'19 (Oral)** [Paper](https://arxiv.org/abs/2101.06679)



> ***ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst*** **RSS'19** [Paper](https://arxiv.org/abs/1812.03079)

**Key insight:** It proposes exposing the learner to synthesized data in the form of perturbations to the expert's driving, which creates interesting situations such as collisions and/or going off the road. Rather than purely imitating all data, it augment the imitation loss with additional losses that **penalized undesirable events and encourage progress**. 

**Method:** The AgentRNN is unrolled at training time for a fixed number of iterations, and the losses described below are **summed together over the unrolled iterations.**



> ***End-to-end Driving via Conditional Imitation Learning*** **ICRA'18** [Paper](https://arxiv.org/abs/1710.02410)



### Tree Search Related 

> ***LEADER: Learning Attention over Driving Behaviors for Planning under Uncertainty*** **CoRL'22 (Oral)** [Paper](https://arxiv.org/abs/2209.11422)

**Key insight:** POMDPs offer a principled framework for planning under uncertainty. However, **sampling also raises safety concerns by potentially missing critical events.** This paper proposes LEADER, that learns to attend to critical human behaviors during planning. LEADER learns a neural network generator **(by solving a min-max game)** to provide attention over human behaviors, using importance sampling to bias reasoning towards critical events. 

**Method:** Critic network: $C_\phi(b,z,q)$. Attention generator network: $G_\psi(b,z)$. The min-max game: $\min_{q\in Q}\max_{\pi\in\Pi}\hat{V}_\pi(b,z|q)$. $q$ is used to bias the simulation process during the DESPOT rollout, which can help sample over critical events with low happen rate. The rollout value is unbiased by the importance weight to enable correct planning.

**Experiment:** SUMMIT.



> ***Closing the Planning-Learning Loop with Application to Autonomous Driving*** **T-RO'22** [Paper](https://arxiv.org/abs/2101.03834) | [Code](https://github.com/cindycia/lets-drive)

**Key insight:** Two challenges for autonomous driving: scalability and uncertainty (partial observable and complex world model). Current POMDP solvers like DESPOT still stuggles with very long horizons or large action spaces, producing highly sub-optimal decisions. Inspired by AlphaGO-Zero, this paper proposes LeTS-Drive, which integrates planning and learning in a closed loop, taking advantage of both self-supervised (learn from the planner improvement) and reinforcement learning (learn from the environment feedback). 

**Method:** POMDP: $(S,A,Z,T,O,R)$ plans a belief-space policy $\pi:\mathcal B\rightarrow A$. The POMDPs execute by the Bellman's operator: $V(b)=\max_{a\in A}\sum_{s\in S}b(s)R(s,a)+\gamma\sum_{z\in Z}p(z|b,a)V(\tau(b,a,z))$. LeTS-Drive **uses POMDPs to model uncertainties.** Specifically, the state of a scene can be summarized as: physical state of the ego-vehicle $s_c$, physical states of exo-agents $\{s_i\}_{i\in I_{exo}}$, and the hidden states of ego-agents. The observation only consists of ego and exo agent physical states, while the hidden states can only be inferred from the history. If the intention is defined as the target routes for exo-agents, then the belief $b$ is just the probability distribution of the intended routes for the exo-agents.

DESPOT reduces the complexity from $O(|A|^H|Z|^H)$ to $O(|A|^H K)$, LeTS-Drive further reduces the complexity to $O(|A|^D K)$ by the policy and value networks. Typically, LeTS-Drive takes the history BEV images at a belief $b$ **(the n-th order history stands for $b$, or the network impiles $b$ implicitly)** to output the policy/value, just the same as AlphaGo-Zero.

**Experiment:** SUMMIT. Given any urban map supported by the OpenStreetMap, it automatically generates realistic traffic.



> ***KB-Tree: Learnable and Continuous Monte-Carlo Tree Search for Autonomous Driving Planning*** **IROS'21** [Paper](https://ieeexplore.ieee.org/document/9636442)

**Key insight:** Using kernel regression and bayesian optimization to enable MCTS in continuous space.



> ***Driving Maneuvers Prediction Based Autonomous Driving Control by Deep Monte Carlo Tree*** **T-VT'20** [Paper](https://ieeexplore.ieee.org/document/9082903) | [Code](https://github.com/winds-line/deep-MCTS)

**Key insight:** This paper develops a deep MCTS control method for vision-based autonomous driving.

**Method:** The MCTS module is composed of the Vehicle State Prediction network $f_{VS}(\cdot)$ and Action and Value Prediction network $f_{AV}(\cdot)$. The vehicle state prediction module deconv and get the next state image given the current state image and the action.

**Experiment:** Simulator USS.



### Interaction Related

> ***M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction*** **CVPR'22** [Paper](https://arxiv.org/abs/2202.11884) | [Code](https://tsinghua- mars- lab.github.io/M2I/)

**Key insight:** Existing models excel at predicting marginal trajectories for single agents, yet it remains an open probelm to jointly predict scene compliant trajectories over multiple agents. This paper exploits the underlying relations between interacting agents and decouple the joint prediction problem into marginal prediction problems. As causality in driving interaction remains an open problem, it pre-labels the influencer-reactor relation based on a heuristic, and proposes a relation predictor to classify interactive relations at inference time.

**Method:** Focus on two interactive agents: $P(Y|X)=P(Y_I,Y_R)\approx P(Y_I|X)P(Y_R|X,Y_I)$. The model is composed of a relation predictor, marginal predictor, and a conditional predictor. It uses VectorNet as the context encoder, and DenseTNT as the trajectory regressor. It uses the distance between cars and temporal logic to define the GT influencer-reactor pairs and train the relation predictor.

For each marginal/conditional trajectory, it predicts $N$ samples with confidence scores. Then it uses a sample selector to choose $K$ trajectory pairs among all $N^2$ pairs for evaluation.



> ***InterSim: Interactive Traffic Simulation via Explicit Relation Modeling*** **IROS'22** [Paper](https://arxiv.org/abs/2210.14413) | [Code](https://github.com/Tsinghua-MARS-Lab/InterSim)

**Key insight:** Existing approaches learn an agent model from large-scale driving data to simulate realistic traffic scenarios, yet it remains an open question to produce consistent and diverse multi-agent interactive behaviors in crowded scenes.

**Method:** Five step procedure: conflict detection, relation-aware conflict resolution, goal driven trajectory prediction, conflict resolution between environment agents, and one-step simulation.

If the bounding boxes between two agents overlap at any time in the future, a conflict is detected and requires the simulator to update the colliding trajectories for better consistency and realism. Typically, it uses a relation predictor to identify influencer-reactor pairs. If an environment agent is chosen as the reactor, its goal point is reset at the colliding point, and the simulator uses DenseTNT to rollout its new trajectory. The iteration lasts until no environment agents and ego-vehicle are colliding with each other.



### Optimization Related

> ***Comprehensive Reactive Safety: No Need for a Trajectory if You Have a Strategy*** **IROS'22** [Paper](https://arxiv.org/abs/2207.00198)



> ***Autonomous Driving Motion Planning With Constrained Iterative LQR*** **T-IT'19** [Paper](https://ieeexplore.ieee.org/document/8671755)



> ***Tunable and Stable Real-Time Trajectory Planning for Urban Autonomous Driving*** **IROS'15** [Paper](https://ieeexplore.ieee.org/abstract/document/7353382)



### Traditional Planning Algorithms

> ***Path Planning using Neural A\* Search*** **ICML'21** [Paper](https://arxiv.org/abs/2009.07476) | [Code](https://github.com/omron-sinicx/neural-astar)



> ***Sampling-based Algorithms for Optimal Motion Planning*** **IJRR'10** [Paper](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/optreadings/rrtstar.pdf)



> ***Practical Search Techniques in Path Planning for Autonomous Driving*** **AAAI'08** [Paper](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf)



## Reinforcement Learning

### Model-based RL

> ***Mismatched No More: Joint Model-Policy Optimization for Model-Based RL*** **NeurIPS'22** [Paper](https://arxiv.org/abs/2110.02758)

**Key insight:** Models that achieve better training performance are not necessarily better for control: the objective mismatch problem. This paper proposes a single objective for jointly training the model and the policy (**using an augmented reward**), such that updates to either component increases a lower bound on expected return. The resulting algorithm is conceptually similar to a GAN: a classifier distinguishes between real and fake transitions, the model is updated to produce transitions that look realistic, and the policy is updated to avoid states where the model prediction are unrealistic. Typically, its model objective includes an additional value term and the policy objective includes an additional classifier term.



> ***Planning with Diffusion for Flexible Behavior Synthesis*** **ICML'22** [Paper](https://arxiv.org/abs/2205.09991)

**Key insight:** In model-based RL, the learned models may not be well-suited to standard trajectory optimization since they have different objectives. This paper proposes to fold as much of the trajectory optimization pipeline as possible into the modeling problem, such that sampling from the model and planning with it become nearly identical. The core of the paper is planning by iteratively denoising trajectories. Based on this, reinforcement learning can be interpreted as guided sampling, goal-conditioned RL can be interpreted as inpainting.

**Method:** It uses a diffusion model to model the trajectories $p_\theta(\tau)$. The iterative denoising process of a diffusion model lends itself to flexible conditioning by way of sampling from perturbed distributions of the form: $\tilde{p}_\theta(\tau)\propto p_\theta(\tau)h(\tau)$. 

Since decision-making can be anti-causal (conditioned on the future), Diffuser predict all timesteps of a plan concurrently. As the input to diffuser, states and actions form a two-dimension array. The diffusion model is trained similar to that of DDPM: $\mathcal L(\theta)=\mathbb E_{i,\epsilon,\tau^0}[||\epsilon-\epsilon_\theta(\tau^i,i)||^2]$. 

Using $O_t$ to be a binary random variable denoting the optimality of timestep $t$ of a trajectory, with $p(O_t=1)=\exp(r(s_t,a_t))$. The RL problem can be transformed into conditional sampling: $p_\theta(\tau^{i-1}|\tau^i,O_{1:T}=1)\approx\mathcal N(\tau^{i-1};\mu+\Sigma g,\Sigma)$, where $g=\nabla_\tau\log p(O_{1:T}|\tau)|_{\tau=\mu}$.

The goal-conditioned RL, as well as the start state constraint, can be interpreted as the inpainting problem. We hardly change the start/goal state at the end of each denoising step.



### Offline RL

> ***The In-Sample Softmax for Offline Reinforcement Learning*** **ICLR'23 (submitted)** [Paper](https://openreview.net/forum?id=u-RuvyDYqCM)

**Key insight:** The critical challenge of offline RL is the insufficient action-coverage. Growing number of methods attempt to approximate an in-sample max, that only uses actions well-covered by the dataset. This paper highlights a simple fact: it is more straightforward to approximate an **in-sample softmax** using only actions in the dataset in the entropy-regularized setting. For some instances, batch-constrained Q learning uses in-sample max; IQL solution depends on the action distribution not just the support.  In sample softmax relies primarily on sampling from the dataset, which is naturally in-sample, rather than requiring samples from an estimate of $\pi_D$.

**Method:** The soft Bellman optimality equations for maximum-entropy RL use the softmax in place of the max, as the temperature $\tau\rightarrow 0$, softmax (log-sum-exp) approaches the max. Through reformulating the in-sample softmax optimality equation, it proves that $\sum_{a:\pi_D(a|s)>0}e^{q(s,a)/\tau}=\mathbb E_{a\sim\pi_D(\cdot|s)}[e^{q(s,a)/\tau-\log\pi_D(a|s)}]$, where we can **sample data directly from the offline dataset** instead of the approximate in-sample generative model (e.g. CVAE). Besides, a closed-form greedy policy can be derived from the operator, which proves that it can deviate (generalize) much from $\pi_D$, while **just sharing the same support.**



> ***Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*** **arXiv'22** [Paper](https://arxiv.org/abs/2208.06193)

**Key insight:** Simply model the policy $\pi_\theta(a|s)$ as a diffusion model for its expressiveness and is easy to be guided. The training loss composes of two terms: a behavior cloning loss for the common diffusion loss, and a Q function maximization loss for policy improvement.



> ***Know Your Boundaries: The Necessity of Explicit Behavioral Cloning in Offline RL*** **arXiv'22** [Paper](https://arxiv.org/abs/2206.00695) 

**Key insight:** The paper proves that the soft policy iteration with a penalized value function is equivalent to policy iteration regularized by $D_{KL}(\pi(s)||\pi_p(s))$ where $\pi_p(a|s)=\text{softmax}(-p(s,a))$. Based on this, it summarizes the penalty functions of some representative algorithms. It then proposes an ideal penalty function with support set constraint. Afterwards, it proves that methods like BRAC-KL could fail because it prefers actions that are more frequently executed, and other methods use inaccurate proxies (limited expressivity of the generative model) to replace the action probability $\beta(a|s)$. The paper uses an **score based** model for the BC model.



> ***Mildly Conservative Q-Learning for Ofﬂine Reinforcement Learning*** **NeurIPS'22** [Paper](https://arxiv.org/abs/2206.04745) | [Code](https://github.com/dmksjfl/MCQ)

**Key insight:** Existing offline RL methods that penalize the unseen actions or regularizing with the behavior policy are too pessimistic which supresses the generalization of the value function (**Figure 1** is intuitive to see the problems). This paper explores mild but enough conservatism for offline learning while not harming generalization. It proposes Mildly Conservative Q-learning (MCQ), where OOD actions are actively trained by assigning them proper pseudo Q values. MCQ owns several superior properties: 1. the MCB operator is guaranteed to behave better than the behavior policy 2. it owns a tighter lower bound than existing policy constraint or value penalization methods 3. erroneous overestimation will not occur with it.

**Method:** It first proposes the mildly conservative bellman (MCB) operator and shows that no erroneous overestimation will occur with the MCB operator. The basic idea is of the operator is that if the learned policy outputs actions that lie in the support region of the behavior policy $\mu$, then it goes for backup; while if OOD actions are generated, it deliberately replace their value estimates with $\max_{a'\sim\text{Support}(\mu(\cdot|s))}Q(s,a')-\delta$ (the pseudo target value).  

In practice, it is intractable to acquire the maximum action value in the support set. Thus, it fit an empirical behavior policy with supervised learning on the static dataset. Then the pseudo values for the OOD actions are them computed by sampling $N$ actions from the empirical behavior policy and taking maximum over their value evaluation. To avoid modeling the support set constraint, it resorts to **OOD sampling**, let all sampled pseudo actions to fit the batch-constraint $Q$ value (therefore will not negatively affect the evaluation upon in-distribution actions, see **Algorithm 1**).



> ***Bootstrapped Transformer for Offline Reinforcement Learning*** **NeurIPS'22** [Paper](https://arxiv.org/abs/2206.08569) | [Code](https://arxiv.org/abs/2206.08569)

**Key insight:** The paper follows Trajectory Transformer and deals with the insufficient distribution coverage problem of the offline dataset. It proposes Bootstrapped Transformer, which incorporates the idea of bootstrapping and leverages the learned model to self-generate more offline data to further boost the sequence model learning.

**Method:** BooT treats each input trajectory as a sequence and **add reward-to-go** $R_t=\sum_{t'=t}^T\gamma^{t'-t}r_{t'}$ after reward $r_t$ as auxiliary information at each timestep $t$. The objective is the log probability (add the new reward-to-go). Beam search is adopted to maximize the cumulative discounted reward plus reward-to-go estimates.

BooT utilizes self-generated trajectories as auxiliary data to further train the model, which is the general idea of bootstrapping. The trajectory generation is to resample the last $T'$ timesteps and concatenate it to the previous steps of the original trajectory. The generation method can be divided into two classes: autoregressive generation and teacher-forcing generation. The training on the augmented dataset can be divided into two classes: Bootstrap-once and Bootstrap-repeat (use the generated data repeatedly).



> ***A Uniﬁed Framework for Alternating Offline Model Training and Policy Learning*** **NeurIPS'22** [Paper](https://arxiv.org/abs/2210.05922) | [Code](https://github.com/Shentao-YANG/AMPL_NeurIPS2022)

**Key insight:** Offline MBRL algorithms can improve the efficiency and stability of policy learning over the model-free algorithms. However, in most existing offline MBRL algorithms, the learning objectives for the dynamic models and the policies are isolated from each other. This paper addresses the objective mismatch problem by developing an iterative offline MBRL framework, where it maximizes a lower bound of the true expected return, by alternating between dynamical-model training and policy learning.



> ***A Policy-Guided Imitation Approach for Offline Reinforcement Learning*** **NeurIPS'22** [Paper](https://arxiv.org/abs/2210.08323) | [Code](https://github.com/ryanxhr/POR)

**Key insight:** Offline RL methods can be categorized into two types: RL-based and imitation-based. RL-based methods enjoy OOD generalization but suffer from off-policy evaluation problem. Imitation-based methods avoid off-policy evaluation but are too conservative to surpass the dataset. This paper proposes an alternative approach, inheriting the training stability of imitation-style methods while still allowing logical OOD generalization. It decomposes the conventional reward-maximizing policy in offline RL into **a guide-policy and an execute-policy.** The algorithm allows **state-compositionality** (choose the state with the highest value in the dataset) from the dataset, rather than **action-compositionality** (choose the action with the highest value in the dataset, conservative, should take OOD actions to improve), conducted in prior imitation-style methods. It can also adapt to new tasks by changing the guide-policy.

**Method:** The job of the guide-policy is to learn the optimal next state given the current state, and the job of the execute-policy is to learn how different actions can produce different next states, given current states. 

The guide-policy is to guide the execute-policy about which state it should go to. We train a state value function $V(s):\mathcal S\rightarrow \mathbb R$. Since the training of $V$ only uses $(s,s')$ samples in the offline dataset, it doesn't suffer from overestimation since there are no OOD actions involved. To approximate the optimal value function in the dataset, it uses the l2 loss with a different weight using **expectile regression** (similar to implicit QL). It also adds a behavior cloning term for regularizing the guide-policy. Then the guide policy 's learning objective: $\max_\omega \mathbb E_{(s,s')\sim\mathcal D}[V_\phi(g_\omega(s))+\alpha\log g_\omega(s'|s)]$.

The job of the execute-policy is to have a strong generalization ability. It adopts the RL via Supervised Learning framework by conditioning the execute policy on $s'$ that encountered in the dataset. Its learning objective is: $\max_\theta \mathbb E_{(s,a,s')\in\mathcal D}[\log\pi_\theta(a|s,s')]$.



> ***MOReL: Model-Based Offline Reinforcement Learning*** **arXiv'21** [Paper](https://arxiv.org/abs/2005.05951v3) | [Code](https://github.com/SwapnilPande/MOReL)

**Key insight:** The paper proposes MORel, an algorithm for model-based offline RL. The framework consists of two steps: (a) learning a pessimistic MDP using the offline dataset; (b) learning a near-optimal policy in this P-MDP. The leared P-MDP has the property that for any policy, the performance in the real environment is approximateky lower-bounded by the performance in the P-MDP.



> ***Offline Reinforcement Learning with Implicit Q-Learning*** **arXiv'21** [Paper](https://arxiv.org/abs/2110.06169) | [Code](https://github.com/ikostrikov/implicit q learning)

**Key insight:** It proposes a new offline RL method that **never needs to evaluate actions outside the dataset,** but still enables the learned policy to improve substantially over the best behavior in the data through generalization. It approximates the policy improvement step implicitly by treating the state value function as a random variable, with randomness defined by the action, then taking a state conditional **upper expectile** to estimate the value of the best actions in that state. The algorithm alternates between fitting the supper expectile value and backing it up to the Q-function. Then the policy is extracted via advantage-weighted behavior Q-learning, which avoids querying out-of-sample actions.



> ***Online and Offline Reinforcement Learning by Planning with a Learned Model*** **NeurIPS'21 (Spotlight)** [Paper](https://arxiv.org/abs/2104.06294) 

**Key insight:** Alternating the tree search planner to improve the policy and value and the network to learn from it and guide the planner in turn. This iterative operation can help the networks continuously learn from the offline dataset (**Reanalyze**).



> ***Ofﬂine Reinforcement Learning as One Big Sequence Modeling Problem*** **NeurIPS'21** [Paper](https://arxiv.org/abs/2106.02039) | [Code](trajectory-transformer.github.io)

**Key insight:** RL is typically concerned with estimating **stationary policies** or **single-step models.** This paper views RL as a generic sequence modeling problem, with the goal being to produce a sequence of actions that lead to a sequence of high rewards. Typically, it uses a Transformer architecture to model distributions over trajectories and repurposes beam search as a planning algorithm.

**Method:** A trajectory can be discreted: $\tau=(...,s_t^1,s_t^2,...,s_t^N,a_t^1,a_t^2,...,a_t^M,r_t,...)$. It adopts two discretization approaches: uniform and quantile. The objetive is to maximize the log likelihood.

Imitation learning just uses the previous trajectory as condition and uses beam search to find the successive trajectories with the largest probability. Goal-conditioned RL also uses the last state as the goal. Offline RL replaces the log-probabilities as the predicted reward signal during beam search.



> ***Model-Based Offline Planning*** **ICLR'21** [Paper](https://arxiv.org/abs/2008.05556) 

**Key insight:** Learn an ensembly world model and a behavior policy to guide planning. The planning result is the reward-reweighted action sequence.



> ***Off-Policy Deep Reinforcement Learning without Exploration*** **ICML'19** [Paper](https://arxiv.org/abs/1812.02900) | [Code](https://github.com/sfujim/BCQ)

**Key insight:** Using CVAE to model the state-action distribution in the offline dataset, then using approximately in-sample max in a batch to conduct Q-learning.



## Imitation Learning

> ***Planning for Sample Efﬁcient Imitation Learning*** **NeurIPS'22** [Paper](https://arxiv.org/abs/2210.09598) | [Code](https://github.com/zhaohengyin/EfficientImitate)

**Key insight:** Imitation learning is free from many issues with reinforcement learning such as reward design and the exploration hardness. However, current IL struggles to achieve both high performance and high in-environment sample efficiency simultaneously. Behavior cloning does not need in-environment interactions, but it suffers from the covariate shift problem which harms its performance. Adversarial imitation learing turns imitation learning into a **distribution matching** problem. It can achieves better performance on some tasks but it requires a large number of in-environment interactions. This paper proposes EfficientImitate, a **planning-based imitation learning method** that can achieve high in-environment sample efficiency and performance simultaneously. It first **extends AIL into the MCTS-based RL**, then sow the seemingly incompatible two classes of imitation learning algorithms (BC and AIL) can be naturally unified under one framework.

**Method:** Sample efficiency in RL: one line of work finds that the reward signal is not a good data source for representation learning in RL and resorts to SSL or pretrained representations; another line of works focues on a learned model, which is promising for sample-efficient learning (imagine additional rollouts). EI combines both of these.

The AIL algorithm trains a discriminator $D$ between policy samples and the expert samples and uses some form of $D$ such as $-\log(1-D)$ as the reward function. Then some model-free RL algorithms are used to maximize the cumulative reward. So this reward can be directly adapted to MCTS search to ensure **long-term distribution matching.** Typically, to ensure the discriminator generalize to abstract states, it uses the multi-step discriminator loss (train the discriminator with model-based rollout). 

The behavior cloning function is integrated to guide the MCTS search, $\hat{\pi}=\alpha\pi_{BC}+(1-\alpha)\pi$. Leveraging the power of planning, the long-term outcomes of certain BC actions can be calculated. It also uses the multi-step BC objective.



> ***Generative Adversarial Imitation Learning*** **arXiv'16** [Paper](https://arxiv.org/abs/1606.03476)
