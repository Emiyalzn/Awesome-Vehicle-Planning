# Awesome-Vehicle-Planning
This repo contains my interested papers related to Autonomous Vehicle Planning. It is worth noting that these articles are broadly related to the topic in my opinion: papers not directly study about vehicle motion planning (e.g. pure RL, IL) may also be collected in the list. You can also refer to more details of these papers in `notes.md`, which records my summarization for the papers. Welcome pull requests for interesting papers!

## Planning

| Subcategory                     | Paper                                                        | Conference                       | Links                                                        |
| ------------------------------- | ------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------ |
| RL Related                      | Rethinking Closed-loop Training for Autonomous Driving       | ECCV'22                          | [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990259.pdf) |
|                                 | UMBRELLA: Uncertainty-Aware Model-Based Ofﬂine Reinforcement Learning Leveraging Planning | NeurIPS'21 Workshop (Best paper) | [Paper](https://arxiv.org/abs/2110.07067)                    |
|                                 | Offline Reinforcement Learning for Autonomous Driving with Safety and Exploration Enhancement | NeurIPS'21 Workshop              | [Paper](https://arxiv.org/abs/2110.07067)                    |
|                                 | Motion Planning for Autonomous Vehicles in the Presence of Uncertainty Using Reinforcement Learning | IROS'21                          | [Paper](https://arxiv.org/abs/2110.00640)                    |
|                                 | Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning | arXiv'20                         | [Paper](https://arxiv.org/abs/2001.08726) \| [Code](https://github.com/cjy1992/interp-e2e-driving) |
|                                 | Model-free Deep Reinforcement Learning for Urban Autonomous Driving | arXiv'19                         | [Paper](https://arxiv.org/abs/1904.09503) \| [Code](https://github.com/cjy1992/interp-e2e-driving) |
|                                 | Learning to Drive in a Day                                   | arXiv'18                         | [Paper](https://arxiv.org/abs/1807.00412)                    |
| IL Related                      | Guided Conditional Diffsuion for Controllable Traffic Simulation | arXiv'22                         | [Paper](https://arxiv.org/abs/2210.17366)                    |
|                                 | Model-Based Imitation Learning for Urban Driving             | NeurIPS'22                       | [Paper](https://arxiv.org/abs/2210.07729) \| [Code](https://github.com/wayveai/mile) |
|                                 | Hierarchical Model-Based Imitation Learning for Planning in Autonomous Driving | arXiv'22                         | [Paper](https://arxiv.org/abs/2210.09539)                    |
|                                 | ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning | ECCV'22                          | [Paper](https://arxiv.org/abs/2207.07601) \| [Code](https://github.com/OpenPerceptionX/ST-P3) |
|                                 | PlanT: Explainable Planning Transformers via Object-Level Representations | CoRL'22                          | [Paper](https://arxiv.org/abs/2210.14222) \| [Code](https://github.com/autonomousvision/plant) |
|                                 | End-to-End Urban Driving by Imitating a Reinforcement Learning Coach | CVPR'21                          | [Paper](https://arxiv.org/abs/2108.08265) \| [Code](https://github.com/zhejz/carla-roach) |
|                                 | Perceive, Predict, and Plan: Safe Motion Planning Through Interpretable Semantic Representations | ECCV'20                          | [Paper](https://arxiv.org/abs/2008.05930)                    |
|                                 | DSDNet: Deep Structured self-Driving Network                 | ECCV'20                          | [Paper](https://arxiv.org/abs/2008.06041)                    |
|                                 | Jointly Learnable Behavior and Trajectory Planning for Self-Driving Vehicles | IROS'19                          | [Paper](https://arxiv.org/abs/1910.04586)                    |
|                                 | End-to-end Interpretable Neural Motion Planner               | CVPR'19 (Oral)                   | [Paper](https://arxiv.org/abs/2101.06679)                    |
|                                 | ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst | RSS'19                           | [Paper](https://arxiv.org/abs/1812.03079)                    |
|                                 | End-to-end Driving via Conditional Imitation Learning        | ICRA'18                          | [Paper](https://arxiv.org/abs/1710.02410)                    |
| Tree Search Related             | LEADER: Learning Attention over Driving Behaviors for Planning under Uncertainty | CoRL'22 (Oral)                   | [Paper](https://arxiv.org/abs/2209.11422)                    |
|                                 | Closing the Planning-Learning Loop with Application to Autonomous Driving | T-RP'22                          | [Paper](https://arxiv.org/abs/2101.03834) \| [Code](https://github.com/cindycia/lets-drive) |
|                                 | KB-Tree: Learnable and Continuous Monte-Carlo Tree Search for Autonomous Driving Planning | IROS'21                          | [Paper](https://ieeexplore.ieee.org/document/9636442)        |
|                                 | Driving Maneuvers Prediction Based Autonomous Driving Control by Deep Monte Carlo Tree | T-VT'20                          | [Paper](https://ieeexplore.ieee.org/document/9082903) \| [Code](https://github.com/winds-line/deep-MCTS) |
| Interaction Modeling            | M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction | CVPR'22                          | [Paper](https://arxiv.org/abs/2202.11884) \| [Code](https://github.com/Tsinghua-MARS-Lab/M2I) |
|                                 | InterSim: Interactive Traffic Simulation via Explicit Relation Modeling | IROS'22                          | [Paper](https://arxiv.org/abs/2210.14413) \| [Code](https://github.com/Tsinghua-MARS-Lab/InterSim) |
| Optimization Related            | Comprehensive Reactive Safety: No Need for a Trajectory if You Have a Strategy | IROS'22                          | [Paper](https://arxiv.org/abs/2207.00198)                    |
|                                 | Autonomous Driving Motion Planning With Constrained Iterative LQR | T-IT'19                          | [Paper](https://ieeexplore.ieee.org/document/8671755)        |
|                                 | Tunable and Stable Real-Time Trajectory Planning for Urban Autonomous Driving | IROS'15                          | [Paper](https://ieeexplore.ieee.org/abstract/document/7353382) |
| Traditional Planning Algorithms | Path Planning using Neural A* Search                         | ICML'21                          | [Paper](https://arxiv.org/abs/2009.07476) \| [Code](https://github.com/omron-sinicx/neural-astar) |
|                                 | Sampling-based Algorithms for Optimal Motion Planning        | IJRR'10                          | [Paper](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/optreadings/rrtstar.pdf) |
|                                 | Practical Search Techniques in Path Planning for Autonomous Driving | AAAI'08                          | [Paper](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf) |

## Reinforcement Learning

| Subcategory    | Paper                                                        | Conference             | Link                                                         |
| -------------- | ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------ |
| Model-based RL | Mismatched No More: Joint Model-Policy Optimization for Model-Based RL | NeurIPS'22             | [Paper](https://arxiv.org/abs/2110.02758)                    |
|                | Planning with Diffusion for Flexible Behavior Synthesis      | ICML'22                | [Paper](https://arxiv.org/abs/2205.09991)                    |
| Offline RL     | The In-Sample Softmax for Offline Reinforcement Learning     | ICLR'23 (submitted)    | [Paper](https://openreview.net/forum?id=u-RuvyDYqCM)         |
|                | Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning | arXiv'22               | [Paper](https://arxiv.org/abs/2208.06193)                    |
|                | Know Your Boundaries: The Necessity of Explicit Behavioral Cloning in Offline RL | arXiv'22               | [Paper](https://arxiv.org/abs/2206.00695)                    |
|                | Mildly Conservative Q-Learning for Ofﬂine Reinforcement Learning | NeurIPS'22             | [Paper](https://arxiv.org/abs/2206.04745) \| [Code](https://github.com/dmksjfl/MCQ) |
|                | Bootstrapped Transformer for Ofﬂine Reinforcement Learning   | NeurIPS'22             | [Paper](https://arxiv.org/abs/2206.08569) \| [Code](https://arxiv.org/abs/2206.08569) |
|                | A Uniﬁed Framework for Alternating Offline Model Training and Policy Learning | NeurIPS'22             | [Paper ](https://arxiv.org/abs/2210.05922)\|[Code](https://github.com/Shentao-YANG/AMPL_NeurIPS2022) |
|                | A Policy-Guided Imitation Approach for Offline Reinforcement Learning | NeurIPS'22             | [Paper](https://arxiv.org/abs/2210.08323) \| [Code](https://github.com/ryanxhr/POR) |
|                | MOReL: Model-Based Offline Reinforcement Learning            | arXiv'21               | [Paper](https://arxiv.org/abs/2005.05951v3) \| [Code](https://github.com/SwapnilPande/MOReL) |
|                | Offline Reinforcement Learning with Implicit Q-Learning      | arXiv'21               | [Paper](https://arxiv.org/abs/2110.06169) \| [Code](https://github.com/ikostrikov/implicit_q_learning) |
|                | Online and Offline Reinforcement Learning by Planning with a Learned Model | NeurIPS'21 (Spotlight) | [Paper](https://arxiv.org/abs/2104.06294)                    |
|                | Offline Reinforcement Learning as One Big Sequence Modeling Problem | NeurIPS'21             | [Paper](https://arxiv.org/abs/2106.02039) \| [Code](trajectory-transformer.github.io) |
|                | Model-Based Offline Planning                                 | ICLR'21                | [Paper](https://arxiv.org/abs/2008.05556)                    |
|                | Off-Policy Deep Reinforcement Learning without Exploration   | ICML'19                | [Paper](https://arxiv.org/abs/1812.02900) \| [Code](https://github.com/sfujim/BCQ) |

## Imitation Learning

| Paper                                           | Conference | Link                                                         |
| ----------------------------------------------- | ---------- | ------------------------------------------------------------ |
| Planning for Sample Efﬁcient Imitation Learning | NeurIPS'22 | [Paper](https://arxiv.org/abs/2210.09598) \| [Code](https://github.com/zhaohengyin/EfficientImitate) |
| Generative Adversarial Imitation Learning       | arXiv'16   | [Paper](https://arxiv.org/abs/1606.03476)                    |
