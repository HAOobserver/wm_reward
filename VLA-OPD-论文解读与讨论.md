# VLA-OPD：论文解读与讨论

> 论文：**VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation**  
> 作者：Zhide Zhong、Haodong Yan、Junfeng Li、Junjie He、Tianran Zhang、Haoang Li  
> 作者单位：The Hong Kong University of Science and Technology (Guangzhou)  
> 版本：arXiv:2603.26666v1，cs.RO，2026-03-27，16 页  
> 本文档依据本地 [`VLA-OPD.pdf`](./VLA-OPD.pdf)、论文项目页以及后续讨论整理。  
> 当前 v1 没有附录或单独的 supplementary material。  
> 为避免 Cursor 的公式渲染问题，本文所有公式均使用纯文本代码块。

---

## 目录

1. 一句话概括
2. 研究背景：SFT 与在线 RL 的矛盾
3. 关键符号与概念
4. “初始化学生”具体是什么意思
5. Teacher 模型从哪里来
6. VLA-OPD 整体训练流程
7. Phase 1：学生 On-policy Sampling
8. Phase 2：教师 Dense Labeling
9. Phase 3：Reverse-KL Optimization
10. Reverse-KL 奖励与梯度推导
11. Forward-KL、Reverse-KL 与 Hard-CE
12. 学生动作概率会不会越来越集中
13. 强化学习究竟如何调整神经网络参数
14. Group Sampling 与 GRPO 的区别
15. VLA-OPD 与 SFT、RL、DAgger 的关系
16. 为什么可能减轻灾难性遗忘
17. Distill + GRPO 的作用
18. 完整训练伪代码
19. 实验设置
20. LIBERO 主要结果
21. RoboTwin2.0 主要结果
22. 训练效率、遗忘与消融实验
23. 附录与复现信息核查
24. 论文的主要优点
25. 局限与批判性分析
26. 讨论问题速查
27. 最终总结

---

## 1. 一句话概括

VLA-OPD 的核心是：

```text
让学生策略自己在环境中执行并暴露错误，
再让冻结的专家 Teacher
对学生实际访问的每个状态提供动作概率，
最后使用 Reverse-KL 将 Teacher 的纠错能力蒸馏给学生。
```

它组合了两类训练方式的优点：

```text
SFT 的优点：
每个动作都有稠密监督，训练稳定、收敛快

在线 RL 的优点：
训练状态来自学生自己的闭环行为，
能够看到学生的错误状态和恢复状态

VLA-OPD：
学生状态 + Teacher 稠密监督
```

更准确地说，它是一种：

```text
On-policy / Interactive Policy Distillation
```

即“在当前学生策略诱导的状态分布上进行在线策略蒸馏”。

---

## 2. 研究背景：SFT 与在线 RL 的矛盾

### 2.1 VLA 是什么

Vision-Language-Action 模型接收：

```text
视觉观测 + 语言指令 + 可选机器人本体状态
```

输出：

```text
机器人动作或动作 token 序列
```

可写成：

```text
π_theta(a | s)
```

其中：

- `s` 是视觉、语言和机器人状态；
- `a` 是动作；
- `theta` 是 VLA 的可训练参数。

预训练 VLA 通常具有较强的通用语义理解能力，但在具体机器人、具体环境和精细操作任务上仍然需要后训练。

### 2.2 离线 SFT

SFT 使用静态专家示范：

```text
D_demo = {(s, a)}
```

优化：

```text
L_SFT(theta)
= - E_(s,a)~D_demo [log π_theta(a | s)]
```

优点：

- 每个时间步都有专家动作；
- 监督稠密；
- 梯度方差小；
- 收敛快；
- 实现简单。

问题是，训练状态和部署状态不同：

```text
训练：
s ~ d_expert

部署：
s ~ d_student
```

学生一旦产生一个小误差，就可能进入专家轨迹没有覆盖的状态，随后不断积累误差。

这被称为：

- distribution shift；
- covariate shift；
- exposure bias；
- compounding error。

### 2.3 在线 RL

在线强化学习让当前学生自己与环境交互：

```text
a_t ~ π_theta(· | s_t)
s_(t+1) ~ Environment(s_t, a_t)
```

因此训练能够看到学生自己产生的错误状态。

但机器人任务的奖励通常是：

```text
任务成功：1
任务失败：0
```

而且只在 episode 结束时产生。

这带来：

- 奖励稀疏；
- 信用分配困难；
- 策略梯度方差大；
- 大量轨迹全部失败时几乎没有学习信号；
- 在线环境交互成本高。

### 2.4 VLA-OPD 的定位

三种训练方式可以写成：

| 范式 | 状态从哪里来 | 监督信号 | 主要特点 |
|---|---|---|---|
| Offline SFT | 专家策略 | 专家动作，稠密 | 快，但有分布偏移 |
| Online RL | 当前学生 | 环境结果奖励，稀疏 | 鲁棒，但样本效率低 |
| VLA-OPD | 当前学生 | Teacher 动作分布，稠密 | 同时利用学生状态与稠密信号 |

因此，VLA-OPD 真正重要的设计是把两个因素拆开：

```text
谁产生训练状态？
当前学生

谁提供监督？
冻结 Teacher
```

---

## 3. 关键符号与概念

| 符号 | 含义 |
|---|---|
| `s_t` | 第 `t` 步状态，包括视觉观测与语言指令 |
| `a_t` | 第 `t` 步学生实际采样并执行的动作 |
| `π_theta` | 当前正在训练的学生策略 |
| `π_tea` | 冻结的 Teacher 策略 |
| `tau_i` | 第 `i` 条学生 rollout 轨迹 |
| `B` | 一个训练 batch 中的 prompt 数量 |
| `G` | 每个 prompt 采样的轨迹数量 |
| `D_prompt` | 任务指令和初始环境状态集合 |
| `d^π` | 策略 `π` 在环境中诱导的状态访问分布 |
| `r_t^OPD` | Teacher 与学生 log-probability 构成的稠密奖励 |
| `D_KL(P || Q)` | 从分布 `P` 到分布 `Q` 的 KL 散度 |

### 3.1 On-policy 是什么意思

On-policy 指：

```text
训练数据由当前正在优化的策略自己产生。
```

第 `k` 轮中：

```text
a_t ~ π_theta_k(· | s_t)
s_(t+1) ~ Environment(s_t, a_t)
```

因此：

```text
s ~ d^π_theta_k
```

与之相对，off-policy 数据可能来自：

- 专家；
- 旧策略；
- replay buffer；
- 其他行为策略。

在本文中：

```text
普通 SFT：
off-policy，因为状态来自专家轨迹

GRPO：
on-policy，因为当前学生自己执行

VLA-OPD：
on-policy，因为状态和实际动作由当前学生产生
```

虽然 Teacher 会对学生状态提供标签，但 Teacher 不控制环境，因此轨迹仍然是 student on-policy。

### 3.2 On-policy 与 Online 不完全等价

```text
On-policy：
强调行为策略就是当前目标策略

Online：
强调训练期间持续与环境交互
```

两者经常同时出现，但概念上不同。

---

## 4. “初始化学生”具体是什么意思

论文不是从随机参数开始训练学生。

初始化分为两个层次。

### 4.1 从预训练 VLA checkpoint 开始

学生使用：

```text
OpenVLA-OFT
```

作为基础模型。

这个模型已经具有：

- 视觉表征能力；
- 语言指令理解能力；
- 通用机器人动作先验；
- 合法动作格式；
- 一定的跨任务知识。

### 4.2 用目标任务示范做少量 SFT

从预训练参数 `theta_pretrained` 出发：

```text
theta_init
= SFT(theta_pretrained, D_demo)
```

优化目标仍然是：

```text
L_SFT(theta)
= - E_(s,a)~D_demo [log π_theta(a | s)]
```

得到的模型就是论文表格中的：

```text
Student Init.
```

具体设置：

```text
LIBERO：
每个任务只使用 1 条专家轨迹

RoboTwin2.0：
每个任务使用 1,000 条专家轨迹
```

需要注意：

```text
“1-traj”是每个任务 1 条，
不是所有任务总共只使用 1 条。
```

### 4.3 为什么不能从随机策略开始

学生至少需要能够：

- 理解任务语言；
- 看懂基本场景；
- 输出合法机器人动作；
- 完成部分简单步骤；
- 生成有任务意义的轨迹。

如果完全随机：

- 学生可能无法接近目标物体；
- 访问的状态没有任务价值；
- Teacher 虽然可以标注，但学生的 on-policy 分布可能长期停留在无意义区域；
- 真实机器人上还会产生安全问题。

因此初始化相当于：

```text
先让学生掌握基本驾驶，
再让教练纠正学生亲自驾驶时出现的问题。
```

LIBERO 中 Student Init. 平均成功率为 48.9%，说明它不是完全不会做，而是闭环稳定性和错误恢复能力不足。

---

## 5. Teacher 模型从哪里来

### 5.1 本文实验中的 Teacher

论文采用：

```text
Teacher：SimpleVLA-RL
Student：OpenVLA-OFT
```

SimpleVLA-RL 是一个通过强化学习获得较强任务性能的 VLA 策略。

其参考性能：

```text
LIBERO 平均成功率：93.9%
RoboTwin2.0 平均成功率：74.0%
```

Teacher 不是通过 VLA-OPD 算法训练出来的，而是在 VLA-OPD 开始前已经存在。

### 5.2 Teacher 在 OPD 中做什么

Teacher 参数完全冻结。

训练时：

```text
学生执行动作并产生状态 s_t
        ↓
Teacher 读取相同状态 s_t
        ↓
输出 π_tea(· | s_t)
        ↓
Teacher 动作不进入环境
```

Teacher 只负责提供：

- 动作 logits；
- 动作 log-probability；
- 软概率分布。

### 5.3 Teacher 可以来自哪些来源

论文认为实际系统中的 Teacher 可以来自：

- 已有开源专家 checkpoint；
- 通过 RL 训练的旧模型；
- 单任务专用策略；
- 专有机器人策略 API；
- 更大、更慢但性能更强的模型。

### 5.4 隐藏成本

“1-traj”只描述学生初始化，不包括 Teacher 的成本。

Teacher 背后可能已经消耗：

- 大规模专家数据；
- 大量环境 rollout；
- 强化学习训练；
- 大量 GPU 算力。

因此 VLA-OPD 证明的是：

```text
如何高效地把已有强 Teacher 的闭环能力
转移给一个学生模型
```

而不是：

```text
仅凭 1 条示范从零获得 93% 成功率
```

---

## 6. VLA-OPD 整体训练流程

训练过程分为三个阶段：

```text
Phase 1：Student Sampling
当前学生与环境交互，生成 on-policy 轨迹

Phase 2：Teacher Labeling
Teacher 对学生访问的每个状态输出动作分布

Phase 3：Student Optimization
使用 Reverse-KL 奖励更新学生
```

完整闭环是：

```text
少量 SFT 初始化
        ↓
得到 Student Init.
        ↓
当前学生采样 G 条轨迹
        ↓
收集学生正常状态、错误状态、恢复状态
        ↓
冻结 Teacher 对每个状态输出动作分布
        ↓
构造 log π_teacher - log π_student
        ↓
使用策略梯度更新学生
        ↓
重新使用更新后的学生采样
        ↓
重复直到收敛
```

---

## 7. Phase 1：学生 On-policy Sampling

第 `k` 轮训练执行当前学生：

```text
a_t ~ π_theta_k(· | s_t)
s_(t+1) ~ P(· | s_t, a_t)
```

得到：

```text
tau_i
= (s_0, a_0, s_1, a_1, ..., s_T)
```

并构成当轮数据：

```text
D_k = {tau_i}
```

### 7.1 为什么必须由学生执行

如果由 Teacher 执行动作：

```text
a_t ~ π_tea(· | s_t)
```

后续状态会重新接近专家状态分布，无法观察：

- 学生抓偏后的状态；
- 学生推错物体后的状态；
- 学生在恢复过程中的状态；
- 学生策略边界附近的状态。

因此：

```text
学生动作必须真正推动环境，
Teacher 只能提供标签，不能接管。
```

### 7.2 On-policy 如何解决复合误差

以“把杯子放到盘子上”为例：

```text
专家：
正确抓取 → 正确抬起 → 正确放置

学生：
抓取偏 2 cm
    ↓
杯子姿态发生变化
    ↓
进入专家数据未覆盖的状态
    ↓
普通 SFT 不知道如何恢复
```

VLA-OPD 会把这个“抓偏后的状态”保存下来，随后由 Teacher 在这个状态上提供恢复动作分布。

因此，学生的失败会主动生成新的训练数据。

---

## 8. Phase 2：教师 Dense Labeling

对学生轨迹中的每个状态：

```text
q_t(a) = π_tea(a | s_t)
```

Teacher 返回动作分布。

对应关系是：

```text
状态 s_t
├── 学生实际执行动作 a_t
├── 学生概率 π_theta(a_t | s_t)
└── Teacher 概率 π_tea(a_t | s_t)
```

### 8.1 Teacher 动作不执行

学生动作：

```text
a_t
```

真正推动环境。

Teacher 给出的动作或动作分布：

```text
a_hat_t 或 π_tea(· | s_t)
```

仅用于训练。

### 8.2 Dense supervision 是什么意思

标准在线 RL 可能只有：

```text
episode 结束：
成功 = 1
失败 = 0
```

VLA-OPD 则是：

```text
第 1 步：Teacher 信号
第 2 步：Teacher 信号
...
第 T 步：Teacher 信号
```

论文称其为：

```text
token-level dense supervision
```

它避免了从最终失败反推出“到底是哪一步动作有问题”的困难。

### 8.3 Token 与环境时间步的歧义

论文文字强调 token-level，但算法中的 `t` 同时承担轨迹时间步索引。

在实际 VLA 中，一个环境动作可能由多个动作 token 组成：

```text
a_t = (a_t^1, a_t^2, ..., a_t^L)
```

对应：

```text
log π_theta(a_t | s_t)
= Σ_l log π_theta(
    a_t^l
    |
    s_t, a_t^1, ..., a_t^(l-1)
  )
```

当前 v1 没有进一步说明：

- 每个环境步包含多少 action tokens；
- 奖励是在 token 级还是 action chunk 级聚合；
- action tokenizer 和采样温度。

这是复现时需要确认的细节。

---

## 9. Phase 3：Reverse-KL Optimization

VLA-OPD 的目标是，在学生访问的状态上，使学生动作分布接近 Teacher：

```text
maximize J(theta)

J(theta)
= E_s~d^π_theta [
    - D_KL(
        π_theta(· | s)
        ||
        π_tea(· | s)
      )
  ]
```

Reverse-KL 定义为：

```text
D_KL(
  π_theta || π_tea
)

= Σ_a π_theta(a | s)
      log [
        π_theta(a | s)
        /
        π_tea(a | s)
      ]
```

方向是：

```text
Student || Teacher
```

论文将其称为 Reverse-KL。

---

## 10. Reverse-KL 奖励与梯度推导

### 10.1 逐动作奖励

学生从自己的分布采样：

```text
a_t ~ π_theta(· | s_t)
```

定义：

```text
r_t^OPD
= - [
    log π_theta(a_t | s_t)
    -
    log π_tea(a_t | s_t)
  ]

= log π_tea(a_t | s_t)
  -
  log π_theta(a_t | s_t)
```

也就是：

```text
r_t^OPD
= log [
    π_tea(a_t | s_t)
    /
    π_theta(a_t | s_t)
  ]
```

### 10.2 奖励不是只看 Teacher 概率

信号由概率比决定：

```text
π_tea(a|s) > π_theta(a|s)
→ r > 0
→ 提高该动作概率

π_tea(a|s) < π_theta(a|s)
→ r < 0
→ 降低该动作概率

π_tea(a|s) = π_theta(a|s)
→ r = 0
```

因此，一个动作即使是 Teacher 的最高概率动作，如果学生对它的概率已经超过 Teacher，也会收到负信号。

### 10.3 期望奖励等于负 Reverse-KL

对学生动作取期望：

```text
E_a~π_theta [r(a)]

= E_a~π_theta [
    log π_tea(a|s)
    -
    log π_theta(a|s)
  ]

= -D_KL(
    π_theta(·|s)
    ||
    π_tea(·|s)
  )
```

因此：

```text
最大化期望 r
等价于
最小化 Reverse-KL
```

注意：

- 完整 KL 一定大于等于 0；
- 期望的负 KL 一定小于等于 0；
- 但单个采样动作的 `r(a)` 可以为正或负。

### 10.4 为什么要 stop-gradient

论文将奖励中的学生 log-probability detach：

```text
r_t
= stopgrad(
    log π_tea(a_t|s_t)
    -
    log π_theta(a_t|s_t)
  )
```

然后计算：

```text
∇_theta log π_theta(a_t|s_t)
· r_t
```

对于固定状态：

```text
∇_theta [
  -D_KL(π_theta || π_tea)
]

= E_a~π_theta [
    ∇_theta log π_theta(a|s)
    · (
        log π_tea(a|s)
        -
        log π_theta(a|s)
      )
  ]
```

所以把 log-ratio 当作 stop-gradient reward，可以得到固定状态上的 Reverse-KL score-function 梯度。

### 10.5 Group-based 梯度

论文算法写成：

```text
∇J
≈ 1 / (B · G)
  · Σ_j Σ_i Σ_t
      ∇_theta log π_theta(a_t,i | s_t,i)
      · r_t,i^OPD
```

其中：

- `j` 遍历 batch 中的 prompt；
- `i` 遍历每个 prompt 的 `G` 条轨迹；
- `t` 遍历时间步或动作 token。

最后：

```text
theta
← theta + alpha · ∇J
```

### 10.6 对总体状态分布目标的严格解释

论文总体目标写了：

```text
s ~ d^π_theta
```

严格来说，当前动作会影响未来状态，因此完整 MDP 梯度还应考虑：

```text
当前动作
→ 未来状态访问分布
→ 未来 KL 奖励
```

如果把它当作完整 RL 目标，通常需要类似 reward-to-go：

```text
R_t = r_t + r_(t+1) + ...
```

但论文公式 7 只使用当前的 `r_t`，没有把未来 KL 奖励传给当前动作。

因此，最准确的算法理解是：

```text
1. 用当前学生采集一批 on-policy 状态
2. 暂时把这些状态视为固定训练数据
3. 在每个状态上最小化条件动作分布的 Reverse-KL
4. 更新学生
5. 再用新学生重新采集状态
```

它更接近交替式在线蒸馏，而不是对完整状态占用分布求精确策略梯度。

---

## 11. Forward-KL、Reverse-KL 与 Hard-CE

### 11.1 KL 散度

KL 散度定义为：

```text
D_KL(P || Q)
= Σ_a P(a)
      log [
        P(a) / Q(a)
      ]
```

它不是严格意义上的距离，因为：

```text
D_KL(P || Q)
≠ D_KL(Q || P)
```

而且不满足三角不等式。

不同文献对 Forward/Reverse 的命名偶尔不同，因此最可靠的方式是直接看 KL 两边的分布。

在本文中：

```text
Forward-KL
= D_KL(
    Teacher || Student
  )

Reverse-KL
= D_KL(
    Student || Teacher
  )
```

### 11.2 Forward-KL

Forward-KL：

```text
D_KL(
  π_tea || π_theta
)
```

期望由 Teacher 分布计算：

```text
E_a~π_tea
```

因此 Student 必须为 Teacher 有概率的所有动作分配概率。

论文将其描述为：

```text
mode-covering / mass-covering
```

如果 Teacher 在学生 OOD 状态上非常不确定、分布很宽，Forward-KL 会迫使学生覆盖这些不确定尾部。

论文观察到：

- 策略熵快速增大；
- 动作分布过度分散；
- 早期性能出现明显下降；
- 产生 entropy explosion。

### 11.3 Hard-CE

Hard-CE 只使用 Teacher 最大概率动作：

```text
a_star
= argmax_a π_tea(a|s)

L_HardCE
= -log π_theta(a_star | s)
```

它会丢掉 Teacher 的软概率信息。

如果 Teacher 分布是：

```text
动作 A：0.45
动作 B：0.44
动作 C：0.11
```

Hard-CE 只保留：

```text
动作 A：1
其他动作：0
```

可能导致：

- Teacher argmax 在相邻状态间频繁切换；
- 学生追踪不稳定硬标签；
- 动作熵快速下降；
- 过早失去探索能力；
- premature entropy collapse。

### 11.4 Reverse-KL

Reverse-KL：

```text
D_KL(
  π_theta || π_tea
)
```

期望由学生分布计算：

```text
E_a~π_theta
```

它优先检查：

```text
学生当前会执行的动作，
Teacher 是否认可？
```

如果学生对 Teacher 低概率动作分配很高概率，会受到强惩罚。

论文将其描述为：

```text
zero-forcing
mode-seeking
```

### 11.5 三者对比

| 目标 | 方向/标签 | 主要倾向 | 论文观察 |
|---|---|---|---|
| Forward-KL | `Teacher || Student` | 覆盖 Teacher 全部分布 | 熵过高 |
| Hard-CE | Teacher top-1 | 逼近单一硬标签 | 熵过早坍缩 |
| Reverse-KL | `Student || Teacher` | 回避 Teacher 低概率区域 | 熵较稳定 |

---

## 12. 学生动作概率会不会越来越集中

### 12.1 关键不是 Teacher 概率高，而是概率比

奖励是：

```text
r(a)
= log [
    π_tea(a|s)
    /
    π_theta(a|s)
  ]
```

所以不会简单地执行：

```text
只要是 Teacher top-1，
就永远提高该动作概率。
```

### 12.2 数值例子

Teacher：

```text
动作 A：0.6
动作 B：0.4
```

学生初始：

```text
动作 A：0.5
动作 B：0.5
```

采到 A：

```text
r(A)
= log(0.6 / 0.5)
≈ +0.182
```

提高 A。

采到 B：

```text
r(B)
= log(0.4 / 0.5)
≈ -0.223
```

降低 B。

学生会向：

```text
[0.6, 0.4]
```

移动。

### 12.3 如果动作 A 被提高过头

假设学生变成：

```text
动作 A：0.8
动作 B：0.2
```

虽然 A 仍是 Teacher 最高概率动作，但：

```text
r(A)
= log(0.6 / 0.8)
≈ -0.288
```

A 收到负信号。

而：

```text
r(B)
= log(0.4 / 0.2)
≈ +0.693
```

B 收到正信号。

因此更新会把学生拉回 Teacher 分布。

### 12.4 理论稳定点

当：

```text
π_theta(a|s)
= π_tea(a|s)
```

对每个非零概率动作：

```text
r(a)
= log(1)
= 0
```

所以全局最优是：

```text
Student distribution
= Teacher distribution
```

不是：

```text
Student
= one-hot(argmax Teacher)
```

### 12.5 Reverse-KL 中包含熵项

负 Reverse-KL 可以展开为：

```text
-D_KL(Student || Teacher)

= E_Student[
    log π_tea(a|s)
  ]
  + H(Student)
```

其中：

```text
H(Student)
= -E_Student[
    log π_theta(a|s)
  ]
```

因此目标同时包含：

```text
提高 Teacher 认可动作的概率
+
保留一定学生策略熵
```

这解释了为什么 Reverse-KL 不等同于 Hard-CE。

### 12.6 实践中仍可能集中到少数模式

理论最优是匹配 Teacher，但实际仍可能出现 mode concentration：

1. **学生只从自己的分布采样**

   如果某个动作概率已经非常低，它几乎不会被采到，也就很难获得恢复概率的正梯度。

2. **动作序列维度很高**

   单个 token 可能有概率，但完整有效动作序列的联合概率可能非常小。

3. **Teacher 分布有多个相隔很远的模式**

   如果学生模型无法同时表达多个模式，Reverse-KL 可能选择其中一个。

4. **采样数量有限**

   较小 `G` 会产生 Monte Carlo 偏差和高方差。

5. **Teacher 本身很尖锐**

   如果 Teacher 已经接近：

   ```text
   [0.99, 0.01]
   ```

   学生最终自然也会接近确定性策略。

6. **有限训练和优化路径依赖**

   实际模型不会精确达到全局最优。

### 12.7 对论文“mode-seeking”的准确理解

如果 Student 与 Teacher 属于同一个可表达分布族，完全优化后的最优解仍然是：

```text
Student = Teacher
```

如果 Teacher 在某状态上完全均匀：

```text
Teacher = uniform distribution
```

Reverse-KL 的最优解也仍然是均匀分布。

所以 Reverse-KL 不能凭空识别 Teacher 不知道的正确动作。

更准确的说法是：

```text
Reverse-KL 强烈抑制学生在 Teacher 低概率区域的概率质量；
在多峰、受限表达和有限优化条件下，
这种性质可能表现为 mode-seeking。
```

论文使用的“bounded mode-seeking”主要是经验性描述，v1 没有给出普遍熵界或一般收敛定理。

---

## 13. 强化学习究竟如何调整神经网络参数

### 13.1 核心原则

强化学习更新可以概括为：

```text
提高高收益动作的概率，
降低低收益动作的概率。
```

环境本身不需要可微。

梯度通过：

```text
log π_theta(a | s)
```

传回策略网络。

### 13.2 基本 Policy Gradient

最基本策略损失：

```text
L_policy
= - Σ_t A_t
      log π_theta(a_t | s_t)
```

其中：

```text
A_t = G_t - baseline
```

回报：

```text
G_t
= r_t
  + gamma r_(t+1)
  + gamma^2 r_(t+2)
  + ...
```

参数更新：

```text
theta
← theta
  - eta ∇_theta L_policy
```

等价地写成梯度上升：

```text
theta
← theta
  + eta Σ_t
      A_t
      ∇_theta log π_theta(a_t | s_t)
```

方向：

```text
A_t > 0
→ 提高该动作概率

A_t < 0
→ 降低该动作概率

A_t = 0
→ 没有明确更新
```

### 13.3 梯度如何传入 VLA

假设动作 logits：

```text
z = W h + b
```

其中：

- `h` 是 VLA 隐藏表示；
- `W,b` 是动作输出层。

softmax 得到：

```text
π_theta(a|s)
= softmax(z)
```

策略损失首先产生：

```text
∂L / ∂z
```

再通过链式法则得到：

```text
∂L / ∂W
∂L / ∂b
∂L / ∂h
```

然后继续向前传播到：

- 动作头；
- Transformer；
- 视觉编码器；
- 其他被设置为 trainable 的模块。

Teacher 完全冻结，不接收梯度。

环境转移也不需要反向传播。

### 13.4 一个 softmax 数值例子

学生：

```text
动作 A：0.4
动作 B：0.6
```

Teacher：

```text
动作 A：0.8
动作 B：0.2
```

若采到 A：

```text
r(A)
= log(0.8 / 0.4)
= log 2
≈ +0.693
```

softmax 的 log-probability 梯度：

```text
∂ log p(A) / ∂ z_A
= 1 - p(A)
= 0.6

∂ log p(A) / ∂ z_B
= -p(B)
= -0.6
```

乘奖励：

```text
A logit 梯度
≈ +0.693 × 0.6
≈ +0.416

B logit 梯度
≈ -0.693 × 0.6
≈ -0.416
```

梯度上升会：

```text
提高 A logit
降低 B logit
```

若采到 B：

```text
r(B)
= log(0.2 / 0.6)
≈ -1.099
```

负奖励会降低 B 的概率。

两类样本的期望更新都把学生推向 Teacher。

### 13.5 PPO / GRPO 为什么使用概率比

在线 RL 通常保存采样时的旧策略：

```text
π_old
```

计算：

```text
rho_t
= π_theta(a_t | s_t)
  /
  π_old(a_t | s_t)
```

PPO/GRPO 使用裁剪：

```text
maximize min(
  rho_t A_t,
  clip(rho_t, 1-epsilon, 1+epsilon) A_t
)
```

目的是避免一次更新把策略改变过大。

### 13.6 GRPO 的 group relative advantage

对同一个任务采样 `G` 条轨迹：

```text
tau_1, ..., tau_G
```

得到结果奖励：

```text
R_1, ..., R_G
```

优势可以写成：

```text
A_i
= (
    R_i - mean(R_1,...,R_G)
  )
  /
  (
    std(R_1,...,R_G) + epsilon
  )
```

成功轨迹获得正 advantage，失败轨迹获得负 advantage。

如果所有轨迹都失败：

```text
R = [0,0,0,0,0,0,0,0]
```

那么：

```text
A ≈ [0,0,0,0,0,0,0,0]
```

这一轮几乎没有学习信号。

这正是稀疏奖励 RL 在机器人任务上样本效率低的重要原因。

### 13.7 VLA-OPD 如何更新

VLA-OPD 把环境结果奖励替换为：

```text
r_t^OPD
= log π_tea(a_t|s_t)
  -
  log π_theta(a_t|s_t)
```

对应损失可以理解为：

```text
L_OPD
= - Σ_t
      stopgrad(r_t^OPD)
      log π_theta(a_t|s_t)
```

因此：

```text
环境 RL：
奖励来自成功/失败

VLA-OPD：
奖励来自 Teacher 与 Student 的概率比
```

两者最终都通过：

```text
∇ log π_theta(a|s)
```

更新神经网络参数。

---

## 14. Group Sampling 与 GRPO 的区别

VLA-OPD 主实验中：

```text
batch size = 64
G = 8
```

对每个 prompt 采样 `G` 条轨迹。

增加 `G` 可以：

- 更好地近似学生轨迹期望；
- 平均环境随机性；
- 降低 Monte Carlo 方差；
- 减少偶然单条轨迹对更新的影响。

但纯 Distill 阶段并不是标准 GRPO。

### 14.1 纯 Distill

```text
不使用环境成功奖励
不计算组内结果排名
不训练 Critic
不使用组相对 outcome advantage
直接平均 token-level Reverse-KL 梯度
```

### 14.2 GRPO

```text
使用环境成功/失败奖励
计算组内相对 advantage
使用新旧策略概率比
通常进行 PPO-style clipping
```

论文中的：

```text
Distill + GRPO
```

表示先完成 VLA-OPD 蒸馏，再使用真正的稀疏奖励 GRPO 继续优化。

---

## 15. VLA-OPD 与 SFT、RL、DAgger 的关系

### 15.1 与 SFT

相同点：

- 都使用专家/Teacher 信息；
- 都具有稠密动作监督；
- 都不依赖最终任务奖励进行主要更新。

不同点：

```text
SFT：
在专家状态上训练

VLA-OPD：
在学生状态上训练
```

### 15.2 与在线 RL

相同点：

- 当前学生自己 rollout；
- 数据来自学生诱导的状态分布；
- 使用 `∇log π` 形式更新。

不同点：

```text
在线 RL：
奖励来自环境结果

VLA-OPD：
奖励来自 Teacher log-probability
```

### 15.3 与 DAgger

DAgger 的基本过程：

```text
学生执行
→ 专家标注学生状态
→ 聚合数据
→ 行为克隆
```

VLA-OPD：

```text
学生执行
→ Teacher 输出学生状态上的软分布
→ Reverse-KL
→ 在线更新
```

主要区别：

| DAgger 常见实现 | VLA-OPD |
|---|---|
| 专家 top-1 硬动作 | Teacher 完整软分布 |
| Hard-CE | Reverse-KL |
| 常聚合成监督数据集 | 每轮使用当前学生状态 |
| 行为克隆形式 | sampled policy-gradient 形式 |

因此可以把 VLA-OPD 理解为：

```text
面向 VLA 的软标签 Reverse-KL DAgger
```

---

## 16. 为什么可能减轻灾难性遗忘

论文认为离线 SFT 容易遗忘，因为：

```text
模型被强迫反复拟合一份固定、狭窄、
可能与预训练能力分布割裂的数据集
```

这可能要求较大的参数移动，覆盖原有知识。

VLA-OPD 的更新状态来自：

```text
s ~ d^π_theta
```

也就是学生当前自然访问的行为流形。

作者称其为：

```text
gentle alignment
```

直觉上：

- 不强迫模型拟合完全割裂的固定状态分布；
- 更新围绕当前策略行为展开；
- 学生策略逐步变化，训练状态也逐步跟随；
- 参数调整可能更平缓。

但需要注意：

- On-policy 本身不必然防止遗忘；
- Reverse-KL 也不保证参数变化一定小；
- 论文没有给出参数距离、遗忘上界或理论证明；
- 当前证据主要来自 4 个 held-out task。

所以更准确的结论是：

```text
在论文实验中，VLA-OPD 比离线 SFT
表现出更好的 unseen-task 保留；
但尚未证明一般意义上的 continual learning 保证。
```

---

## 17. Distill + GRPO 的作用

纯蒸馏主要复制 Teacher 能力，因此通常受到 Teacher 上限约束。

论文采用：

```text
脆弱 Student Init.
        ↓
VLA-OPD 稠密蒸馏
        ↓
快速获得接近 Teacher 的强策略
        ↓
GRPO 稀疏奖励微调
        ↓
针对最终任务成功率继续优化
```

为什么先蒸馏再 RL：

1. 初始学生成功率低，直接 GRPO 很难采到成功轨迹。
2. 大量 group 全部失败时，相对 advantage 接近 0。
3. OPD 先提供逐步 Teacher 信号。
4. 学生成功率提高后，GRPO 更容易获得有差异的成功/失败样本。
5. 最终环境奖励可以继续修正 Teacher 蒸馏的性能上限。

LIBERO 平均结果：

```text
Student Init.       48.9%
VLA-OPD Distill     87.4%
Distill + GRPO      93.4%
Teacher             93.9%
```

---

## 18. 完整训练伪代码

```text
输入：
  预训练学生 π_theta
  冻结 Teacher π_tea
  prompt 数据集 D_prompt
  batch size B
  group size G
  learning rate alpha

步骤 0：学生初始化
  使用少量专家轨迹对预训练学生做 SFT

while 未收敛：

  1. 从 D_prompt 抽取 B 个任务/初始状态

  2. 对每个任务：

       使用当前学生采样 G 条轨迹

       对第 i 条轨迹：

         for t = 0 ... T：

           学生观察 s_t

           a_t ~ π_theta(· | s_t)

           在环境中执行 a_t

           保存：
             s_t
             a_t
             log π_theta(a_t | s_t)

           把相同 s_t 输入冻结 Teacher

           获得：
             log π_tea(a_t | s_t)

           计算：
             r_t
             = log π_tea(a_t | s_t)
               - log π_theta(a_t | s_t)

  3. 计算：

       ∇J
       ≈ 1/(B·G)
         Σ_j Σ_i Σ_t
           ∇ log π_theta(a_t,i | s_t,i)
           · stopgrad(r_t,i)

  4. 更新：

       theta
       ← theta + alpha ∇J

  5. 使用更新后的学生重新采集下一轮轨迹

可选：
  蒸馏完成后使用环境成功奖励进行 GRPO
```

---

## 19. 实验设置

### 19.1 Benchmark

论文使用：

1. **LIBERO**

   - 单臂操作；
   - Spatial；
   - Object；
   - Goal；
   - Long；
   - 学生每任务 1 条示范初始化。

2. **RoboTwin2.0**

   - 双臂操作；
   - 选择 4 个代表性任务；
   - 包含短、中、长时程；
   - 学生每任务 1,000 条示范初始化。

### 19.2 模型

```text
Student：
OpenVLA-OFT

Teacher：
SimpleVLA-RL
```

### 19.3 对比方法

- Student Init.；
- Offline SFT；
- GRPO；
- Octo；
- OpenVLA；
- Nora；
- π0 + FAST；
- π0；
- RDT；
- Teacher reference。

### 19.4 主实验采样设置

```text
batch size = 64
group size G = 8
```

### 19.5 两种论文方法

```text
Ours (Distill)
只进行 VLA-OPD

Ours (Distill + GRPO)
先进行 VLA-OPD，再进行 GRPO
```

---

## 20. LIBERO 主要结果

### 20.1 原始成功率

单位：成功率 `%`。

| 方法 | 数据条件 | Spatial | Object | Goal | Long | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| SimpleVLA-RL Teacher | Teacher | 94.2 | 96.1 | 94.6 | 90.7 | 93.9 |
| Octo | 50 demos/task | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 50 demos/task | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| Nora | 50 demos/task | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 |
| π0 + FAST | 50 demos/task | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| OpenVLA-OFT Student Init. | 1 demo/task | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| VLA-OPD Distill | 1 demo/task + Teacher | 84.3 | 93.8 | 92.5 | 78.9 | 87.4 |
| VLA-OPD Distill + GRPO | 1 demo/task + Teacher + RL | 93.4 | 95.3 | 94.5 | 90.2 | 93.4 |

### 20.2 同一学生管线内的提升

| Suite | Student Init. | Distill | Distill 提升 | Distill + GRPO | Teacher | 最终距 Teacher |
|---|---:|---:|---:|---:|---:|---:|
| Spatial | 63.6 | 84.3 | +20.7 pp | 93.4 | 94.2 | -0.8 pp |
| Object | 54.9 | 93.8 | +38.9 pp | 95.3 | 96.1 | -0.8 pp |
| Goal | 59.6 | 92.5 | +32.9 pp | 94.5 | 94.6 | -0.1 pp |
| Long | 17.3 | 78.9 | +61.6 pp | 90.2 | 90.7 | -0.5 pp |
| Avg. | 48.9 | 87.4 | +38.5 pp | 93.4 | 93.9 | -0.5 pp |

`pp` 表示百分点。

从 48.9% 到 87.4% 是：

```text
绝对提升：
87.4 - 48.9
= 38.5 个百分点

相对提升：
38.5 / 48.9
≈ 78.7%
```

项目页写“+38.5%”容易引起歧义，更准确的表达是：

```text
+38.5 percentage points
```

### 20.3 关闭 Teacher 差距

Student Init. 到 Teacher：

```text
93.9 - 48.9
= 45.0 pp
```

纯 Distill 改善：

```text
87.4 - 48.9
= 38.5 pp
```

关闭的 Teacher 差距：

```text
38.5 / 45.0
≈ 85.6%
```

Distill + GRPO：

```text
93.4 - 48.9
= 44.5 pp
```

关闭：

```text
44.5 / 45.0
≈ 98.9%
```

### 20.4 最有意义的数据

最有意义的不是跨论文榜单，而是同一 Student Init. 管线内：

```text
48.9 → 87.4 → 93.4
```

原因是跨方法比较可能混合：

- 不同骨干；
- 不同预训练数据；
- 不同实现；
- 不同计算预算；
- 不同评估配置。

同一学生初始化到 Distill 的变化更能直接反映 VLA-OPD 的效果。

### 20.5 长时程任务

LIBERO-Long：

```text
17.3 → 78.9
+61.6 pp
```

这是四个 suite 中提升最大的一项。

它与论文动机吻合：

```text
任务越长，
小误差越容易积累，
恢复状态越重要，
on-policy Teacher correction 的价值越大。
```

---

## 21. RoboTwin2.0 主要结果

### 21.1 原始结果

单位：成功率 `%`。

| 方法 | Pick Dual Bottles | Place Empty Cup | Handover Block | Stack Bowls Two | Avg. |
|---|---:|---:|---:|---:|---:|
| SimpleVLA-RL Teacher | 68.3 | 94.2 | 57.8 | 75.8 | 74.0 |
| π0 | 50.0 | 60.0 | 39.0 | 53.0 | 50.5 |
| RDT | 18.0 | 42.0 | 26.0 | 42.0 | 32.0 |
| OpenVLA-OFT Student Init. | 29.7 | 77.3 | 33.1 | 40.6 | 45.2 |
| VLA-OPD Distill | 66.4 | 90.6 | 52.3 | 75.0 | 71.1 |

任务时程：

```text
Pick Dual Bottles：Short
Place Empty Cup：Medium
Handover Block：Long
Stack Bowls Two：Long
```

### 21.2 逐任务提升

| 任务 | Student Init. | Distill | 提升 | Teacher | 距 Teacher |
|---|---:|---:|---:|---:|---:|
| Pick Dual Bottles | 29.7 | 66.4 | +36.7 pp | 68.3 | -1.9 pp |
| Place Empty Cup | 77.3 | 90.6 | +13.3 pp | 94.2 | -3.6 pp |
| Handover Block | 33.1 | 52.3 | +19.2 pp | 57.8 | -5.5 pp |
| Stack Bowls Two | 40.6 | 75.0 | +34.4 pp | 75.8 | -0.8 pp |
| Avg. | 45.2 | 71.1 | +25.9 pp | 74.0 | -2.9 pp |

### 21.3 关闭 Teacher 差距

初始 Teacher 差距：

```text
74.0 - 45.2
= 28.8 pp
```

Distill 改善：

```text
71.1 - 45.2
= 25.9 pp
```

关闭：

```text
25.9 / 28.8
≈ 89.9%
```

### 21.4 如何理解

RoboTwin2.0 学生已经使用每任务 1,000 条轨迹做 SFT，但平均仍只有 45.2%。

这说明：

```text
更多离线示范
不一定自动解决闭环分布偏移和双臂协调错误恢复。
```

VLA-OPD 将其提升到 71.1%，接近 Teacher 的 74.0%。

不过 Handover Block 仍有 5.5 pp 差距，说明复杂长时程双臂协调并未被完全蒸馏。

---

## 22. 训练效率、遗忘与消融实验

### 22.1 训练效率

LIBERO-Object：

```text
VLA-OPD Distill
约 10 个训练 step 内超过 90%
```

LIBERO-Long：

```text
VLA-OPD Distill
约 50 step 达到接近 80%

GRPO
约 150 step 才达到相近结果
```

论文称其约为：

```text
3× speedup
```

但这个结论按 optimizer step 计算。

一个 OPD step 还包含：

- 多条环境 rollout；
- 每个状态的 Teacher 推理；
- Student 与 Teacher logits；
- Group sampling。

论文没有给出：

- environment frames；
- Teacher query 数；
- GPU hours；
- wall-clock；
- FLOPs。

因此：

```text
3× 是更新步数加速，
不能直接等价为 3× 墙钟时间或 3× 总算力节省。
```

### 22.2 灾难性遗忘

论文在 seen tasks 上后训练，并评估 4 个 held-out unseen tasks：

```text
2 个 Object
2 个 Spatial
```

图 3 表明：

- Offline SFT 的 seen 成功率提高时，unseen 性能显著下降；
- Object unseen task 中部分结果接近 0；
- RL 与 VLA-OPD 更好地保留 unseen 能力；
- VLA-OPD 在多个轴上达到或接近 RL。

局限：

- 没有数值表；
- 没有均值与标准差；
- 没有置信区间；
- 只有 4 个 unseen task；
- 没有长任务序列；
- 不能证明一般 continual learning 能力。

### 22.3 Reverse-KL / Forward-KL / Hard-CE 消融

实验任务：

```text
RoboTwin2.0
Beat Block Hammer
```

论文图 4 观察：

```text
Reverse-KL：
成功率稳定提高，最终最高

Forward-KL：
早期出现超过 50% 的 performance valley

Hard-CE：
恢复较差，最终停在最低平台
```

对应 actor entropy：

```text
Forward-KL：
熵快速升高，entropy explosion

Hard-CE：
熵快速降低，premature collapse

Reverse-KL：
熵保持在中间、相对稳定
```

需要谨慎：

- 只在一个任务上展示；
- 没有多 seed；
- 没有逐 checkpoint 数值表；
- “bounded”没有形式化理论界。

### 22.4 Group Size 消融

设置：

```text
LIBERO-Object
batch size = 32
G ∈ {2,4,8}
```

结果：

| G | 论文报告 |
|---:|---|
| 2 | 最终超过 80%，没有崩溃 |
| 4 | 稳定提升，介于 G=2 与 G=8 |
| 8 | 最平滑，最终约 89% |

结论：

- 更大 `G` 能降低 Monte Carlo 方差；
- `G=2` 也具有可用信号；
- 小 `G` 可以减少 rollout 与 Teacher 推理成本。

但实验没有在相同总 rollout 预算下比较：

```text
G=8 每个更新本来就比 G=2 使用更多轨迹。
```

---

## 23. 附录与复现信息核查

### 23.1 当前论文没有附录

本地 PDF 与 arXiv v1 均为 16 页：

```text
第 1-13 页：正文
第 13-16 页：参考文献
```

没有：

- Appendix；
- Supplementary Material；
- 单独 supplementary PDF；
- 完整超参数表。

### 23.2 正文明确给出的信息

| 项目 | 设置 |
|---|---|
| Student | OpenVLA-OFT |
| Teacher | SimpleVLA-RL，冻结 |
| LIBERO 初始化 | 1 demo/task |
| LIBERO 完整数据基线 | 50 demos/task |
| RoboTwin2.0 初始化 | 1,000 demos/task |
| 主实验 batch size | 64 |
| 主实验 group size | 8 |
| Group 消融 batch size | 32 |
| Group 消融 | `G={2,4,8}` |
| 纯 Distill 环境奖励 | 不使用 |
| Teacher 监督 | 学生访问状态上的动作 logits |

### 23.3 当前缺失的复现信息

论文 v1 没有清楚报告：

1. 实际 learning rate；
2. optimizer；
3. scheduler；
4. weight decay；
5. gradient clipping；
6. 混合精度设置；
7. 具体冻结和更新哪些学生层；
8. 动作 tokenization；
9. sampling temperature；
10. rollout horizon；
11. episode 截断方式；
12. 每个 checkpoint 的评测 episode 数；
13. 随机种子；
14. 重复实验次数；
15. 标准差或置信区间；
16. Teacher query 数；
17. environment steps；
18. GPU 型号和数量；
19. wall-clock；
20. Teacher 训练预算和 checkpoint 选择过程；
21. 真实机器人安全机制；
22. Teacher logits 的数值稳定处理；
23. 是否对极小 Teacher 概率进行 clipping。

Algorithm 1 把学习率 `alpha` 作为输入，但没有给出具体数值。

### 23.4 Code 状态

截至 2026-07-15，项目页仍标注：

```text
Code (Coming Soon)
```

所以目前：

- 可以理解算法；
- 可以根据公式实现近似版本；
- 但不能确认作者的精确工程细节；
- 不能完全复现实验配置。

---

## 24. 论文的主要优点

### 24.1 把“状态来源”和“监督来源”拆开

论文最有价值的洞见是：

```text
On-policy 不必与稀疏奖励绑定。
```

可以同时使用：

```text
学生当前状态分布
+
Teacher 稠密监督
```

### 24.2 直接针对 exposure bias

方法不是间接正则化学生，而是直接：

```text
采集学生走偏后的状态
→ 在这些状态上训练恢复动作
```

### 24.3 长时程任务收益明显

LIBERO-Long：

```text
17.3 → 78.9
```

是最符合方法动机的结果。

### 24.4 蒸馏和 RL 可以组合

OPD 适合快速 warm start，GRPO 适合在成功轨迹已经较多后继续按任务结果优化。

### 24.5 不需要 Critic

纯 Distill 阶段使用 Teacher log-ratio，不需要额外训练价值网络，减少了大模型后训练的显存负担。

---

## 25. 局限与批判性分析

### 25.1 Teacher 成本被转移而不是消失

学生可能只用 1 条示范初始化，但 Teacher 已经是强 RL 模型。

总系统成本包括：

- Teacher 训练；
- Teacher 数据；
- Teacher 环境交互；
- Teacher 在线推理；
- 学生 rollout。

### 25.2 Teacher 在学生 OOD 状态上未必可靠

VLA-OPD 专门采集学生失败状态，但这些状态可能也超出 Teacher 的训练分布。

如果 Teacher：

- 给错动作；
- 高熵犹豫；
- 校准不良；
- 在视觉异常状态中失效；

蒸馏会传递这些问题。

Reverse-KL 不能创造 Teacher 没有的知识。

### 25.3 Reverse-KL 理论表述偏强

论文把 Reverse-KL 描述为能够：

- 过滤 Teacher epistemic uncertainty；
- 保留动作多样性；
- bounded mode-seeking。

但：

```text
若 Student 可完整表达 Teacher，
Reverse-KL 全局最优仍是 Student = Teacher。
```

Teacher 完全均匀时，Reverse-KL 也会匹配均匀分布。

现有证据主要是一个任务上的训练动态，不是一般性定理。

### 25.4 样本效率口径不完整

论文主要按训练 step 比较。

但没有报告：

- 每 step 环境帧；
- 每 step Teacher queries；
- Teacher 推理成本；
- 总 GPU 小时；
- 墙钟时间。

所以不能据此得出完整系统成本降低 3 倍。

### 25.5 “1-demo”容易被误读

更准确的实验条件是：

```text
学生 1-demo 初始化
+
已有强 Teacher
+
大量 on-policy 学生轨迹
```

而不是只使用一条数据。

### 25.6 与跨论文基线不是严格受控比较

Octo、OpenVLA、Nora、π0 + FAST 可能使用不同：

- 骨干；
- 预训练；
- 动作表示；
- 数据；
- 超参数；
- 训练预算。

最可信的对比仍然是同一 OpenVLA-OFT Student Init. 前后。

### 25.7 遗忘证据有限

只有：

- 4 个 held-out task；
- 一次微调过程；
- 定性散点图。

缺少：

- 长序列 continual learning；
- 多次任务切换；
- 更多能力维度；
- 参数距离；
- replay 对比；
- 统计显著性。

### 25.8 没有真实机器人实验

所有主要结果来自仿真 benchmark。

真实机器人还需要考虑：

- 学生失败 rollout 的安全性；
- 碰撞；
- 人类接管；
- 环境重置；
- Teacher 延迟；
- 相机噪声；
- sim-to-real gap。

### 25.9 Teacher API 需要 logits

Reverse-KL 需要：

```text
π_tea(a | s)
或
log π_tea(a | s)
```

只返回 top-1 动作的闭源 API 无法直接支持该目标。

### 25.10 可能存在低概率模式难以恢复

因为 Reverse-KL 从学生分布采样：

```text
a ~ π_student
```

如果某个有效动作的学生概率已经非常小，它可能很久都不会被采到，从而难以恢复。

这也是 Reverse-KL 实际表现为 mode-seeking 的一个原因。

---

## 26. 讨论问题速查

### 26.1 “初始化学生”是什么意思

```text
预训练 OpenVLA-OFT
    ↓
少量目标任务 SFT
    ↓
Student Init.
```

不是随机初始化，也不是复制 Teacher 参数。

### 26.2 On-policy 是什么意思

```text
当前正在训练的 Student
自己执行动作并产生训练轨迹。
```

Teacher 只标注，不接管环境。

### 26.3 Teacher 从哪里来

本文使用预先训练好的 SimpleVLA-RL。

它在 OPD 中冻结，Teacher 的训练成本不包含在“1-traj student initialization”中。

### 26.4 Forward-KL 是 KL 散度吗

是。

本文中：

```text
Forward-KL
= D_KL(Teacher || Student)

Reverse-KL
= D_KL(Student || Teacher)
```

KL 不对称，因此方向会改变优化行为。

### 26.5 高 Teacher 概率动作会不会一直增大

不会只因为 Teacher 概率高就一直增大。

信号看：

```text
Teacher probability / Student probability
```

若学生概率已经超过 Teacher，该动作会获得负信号。

理论稳定点是：

```text
Student = Teacher
```

但有限采样和高维序列仍可能造成实际模式集中。

### 26.6 强化学习如何调整参数

核心是：

```text
reward / advantage
×
∇ log π_theta(a|s)
```

正 advantage 提高动作概率，负 advantage 降低动作概率，然后通过普通反向传播更新 VLA。

### 26.7 VLA-OPD 属于强化学习吗

它使用：

- on-policy rollout；
- score-function policy gradient；
- sampled actions。

但监督不是环境回报，而是 Teacher 分布。

所以更准确的名称是：

```text
on-policy policy distillation
或
interactive imitation learning
```

### 26.8 为什么还要 Distill + GRPO

OPD 快速把学生拉到 Teacher 附近；随后 GRPO 使用真实任务成功奖励继续优化，避免纯蒸馏完全受 Teacher 上限限制。

---

## 27. 最终总结

VLA-OPD 的完整逻辑是：

```text
1. 从预训练 OpenVLA-OFT 开始

2. 用少量专家示范做 SFT
   得到具有基本能力但不稳定的 Student Init.

3. 让当前 Student 自己在环境中执行
   采集正常状态、失败状态和恢复状态

4. 冻结 SimpleVLA-RL Teacher
   对每个 Student 状态输出动作概率

5. 对 Student 采样动作计算：

   r
   = log π_teacher
     - log π_student

6. 使用：

   ∇ log π_student · r

   更新 Student

7. 重复采样与蒸馏

8. 可选地追加 GRPO
   使用真实任务成功奖励继续优化
```

论文最可信的实验结论是：

```text
LIBERO：
48.9 → 87.4 → 93.4

RoboTwin2.0：
45.2 → 71.1
```

其中：

- LIBERO 纯 Distill 关闭约 85.6% 的 Teacher 差距；
- Distill + GRPO 关闭约 98.9%；
- RoboTwin2.0 纯 Distill 关闭约 89.9%；
- LIBERO-Long 获得最大的 +61.6 pp 提升。

最值得保留的研究洞见是：

```text
“On-policy”不必等于“稀疏奖励 RL”。

可以让学生产生真实部署状态，
同时使用 Teacher、规则或其他模型
提供稠密学习信号。
```

对论文应保持的主要谨慎是：

```text
它高效转移了已有强 Teacher 的能力，
但没有消除 Teacher 的训练成本；

Reverse-KL 的经验表现很好，
但“过滤不确定性”和“bounded mode-seeking”
还不是一般性理论保证；

实验主要来自仿真，
且缺少附录、代码、随机种子、
误差条和完整计算成本。
```

因此，合理的总体评价是：

> VLA-OPD 是一个思路清晰、实验增益显著、工程价值较高的 VLA 后训练框架。它最重要的贡献是“学生状态分布上的稠密 Teacher 监督”；Reverse-KL 是有效的优化选择，但相关理论主张和真实机器人可扩展性仍需要更多证据。

