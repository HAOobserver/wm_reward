# WAM-RL：论文解读与讨论

> 论文：**WAM-RL: World-Action Model Reinforcement Learning with Reconstruction Rewards and Online Video SFT**  
> 作者：Zezhong Qian 等  
> 版本：arXiv:2606.17906v1，2026-06-16，8 页  
> 本文档依据本地 `WAM-RL.pdf` 以及后续讨论整理。

## 1. 一句话概括

WAM-RL 同时后训练 World-Action Model 中的两个模块：

- **Actor** 使用强化学习，让机器人真实执行的视频尽量符合世界模型想象的视频；
- **世界模型** 使用在线成功轨迹做 Video SFT，让其想象逐渐包含更合理的任务过程与恢复行为；
- 使用 latent KL 正则限制世界模型表示漂移，避免 Actor 无法继续理解更新后的 latent。

更精确地说：

> Actor RL 让“现实接近想象”；Online Video SFT 让“想象接近成功的现实”。

---

## 2. 研究背景

### 2.1 什么是 World-Action Model

普通 VLA（Vision-Language-Action）模型通常从视觉、语言和机器人状态直接预测动作。

World-Action Model（WA/WAM）进一步显式预测未来世界状态：

```text
当前观测、任务信息
        ↓
世界模型想象未来视频，并生成 latent
        ↓
Actor 将 latent 翻译为机器人动作
        ↓
环境执行动作
```

论文认为：

- 世界模型承担隐式规划，决定“未来应该发生什么”；
- Actor 更像翻译器，决定“怎样将 latent 计划转化为动作”。

### 2.2 现有方法的问题

已有 WA 模型主要依靠专家演示进行监督训练，存在两个限制：

1. 策略能力受专家数据分布限制；
2. 部署后不能从自己的在线交互中继续改进。

只对 Actor 做 RL 也不一定够：

- 短时程任务中，Actor 可以改善动作精度；
- 长时程任务中，如果世界模型的预测逐渐出错，Actor 无法修复上游错误计划。

因此论文提出同时更新 Actor 和世界模型。

---

## 3. 方法总览

一次在线交互形成两条更新路径：

```text
当前观测
   ↓
世界模型生成预测视频和 latent
   ↓
Actor 根据 latent 生成动作
   ↓
动作在环境中执行
   ↓
得到真实执行视频及 success/fail
   ├──────────────────────────────────────┐
   ↓                                      ↓
预测视频 vs 真实视频                 如果最终成功
   ↓                                      ↓
Reconstruction Reward              Online Video SFT + KL
   ↓                                      ↓
更新 Actor                           更新世界模型
```

需要注意：

- “联合优化”不是使用同一个 RL loss 同时更新两个模块；
- 只有 Actor 接受 policy gradient；
- 世界模型仍然通过视频预测损失进行监督/自监督微调；
- 两者通过在线交互数据形成协同更新闭环。

---

## 4. Flow Matching、ODE 与 Flow-SDE

### 4.1 ODE 是什么

ODE 是 Ordinary Differential Equation，即常微分方程：

$$
\frac{dx}{dt}=v(x,t)
$$

它描述状态 $x$ 随时间的确定性变化速度。

使用 Euler 方法离散化后：

$$
x_{k+1}=x_k+v(x_k,t_k)\Delta t
$$

给定相同初始状态和相同模型，ODE 会产生相同轨迹。

直观类比：

```text
相同起点 + 相同水流 → 相同运动轨迹
```

### 4.2 Flow Matching

Flow Matching 学习一个连续速度场：

$$
\frac{dx_\tau}{d\tau}=v_\theta(x_\tau,\tau)
$$

该速度场把简单噪声分布逐步运输到动作或视频数据分布：

```text
高斯噪声
   ↓
多次 Flow 更新
   ↓
最终动作或视频样本
```

这里的 $\tau$ 是模型内部的生成/去噪时间，不是机器人与环境交互的时间。

### 4.3 SDE 是什么

SDE 是 Stochastic Differential Equation，即随机微分方程：

$$
dx=v(x,t)dt+\sigma dW_t
$$

其中：

- $v(x,t)dt$ 是确定性的变化趋势；
- $dW_t$ 是布朗运动，即微小随机噪声；
- $\sigma$ 控制噪声强度。

离散化后可以理解为：

$$
x_{k+1}
=x_k+v(x_k,t_k)\Delta t
+\sigma\sqrt{\Delta t}\epsilon_k,
\qquad
\epsilon_k\sim\mathcal N(0,I)
$$

直观类比：

```text
固定水流 + 随机风浪
相同起点 → 每次可能得到不同轨迹
```

### 4.4 Flow-SDE

Flow-SDE 是把原来确定性的 Flow ODE 改造成随机生成过程：

$$
dx_\tau
=v_\theta(x_\tau,\tau)d\tau
+\sigma dW_\tau
$$

每个离散生成步骤都可以表示为高斯转移：

$$
p(x_{\tau-\Delta}\mid x_\tau)
=
\mathcal N\left(
\mu_\theta(x_\tau,\tau),
\sigma^2I
\right)
$$

这样做有两个作用：

1. 在动作生成过程中加入显式探索；
2. 每个生成步骤都有可计算的概率密度，可以使用策略梯度。

严格来说，原始 Flow ODE 从不同初始噪声出发也能产生不同动作。因此“ODE 完全没有随机性”并不准确。更准确的表述是：

> 给定初始噪声后，ODE 的每个中间转移是确定性的；Flow-SDE 则在每个转移中显式加入可计算概率的随机性。

---

## 5. `log p` 在强化学习中的作用

### 5.1 `p` 和 `log p`

$$
p=\pi_\theta(a\mid s)
$$

表示策略在状态 $s$ 下选择动作 $a$ 的概率或概率密度。

$$
\log p=\log\pi_\theta(a\mid s)
$$

是该概率密度的自然对数。

使用对数有三个主要好处：

1. 概率连乘变成对数求和；
2. 避免大量小概率相乘造成数值下溢；
3. 可以利用 log-derivative trick 计算策略梯度。

### 5.2 策略梯度

RL 希望最大化期望奖励：

$$
J(\theta)
=
\mathbb E_{a\sim\pi_\theta}[R]
$$

利用：

$$
\nabla_\theta p
=
p\nabla_\theta\log p
$$

可得到：

$$
\nabla_\theta J
=
\mathbb E[
A(s,a)\nabla_\theta
\log\pi_\theta(a\mid s)
]
$$

其中 $A(s,a)$ 是 advantage：

- $A>0$：该动作表现好，提高其概率；
- $A<0$：该动作表现差，降低其概率；
- $A\approx0$：基本不更新。

常见策略损失为：

$$
L_{\text{policy}}
=
-A\log\pi_\theta(a\mid s)
$$

奖励负责评价动作，`log p` 提供把评价反向传给策略参数的可微入口。

### 5.3 Flow-SDE 中的 `log p`

Flow-SDE 通过多个内部随机步骤生成最终动作：

```text
xK → xK-1 → ... → x1 → 最终动作
```

内部轨迹概率是：

$$
p(\text{trajectory})
=
\prod_k p(x_{k-1}\mid x_k)
$$

取对数后：

$$
\log p(\text{trajectory})
=
\sum_k \log p(x_{k-1}\mid x_k)
$$

论文将其写成：

$$
\log\pi_\theta(a\mid s)
=
\sum_k \log p(x_{k-1}\mid x_k)
$$

高奖励动作会提高对应内部去噪转移的概率，低奖励动作则会降低其概率。

---

## 6. Actor 的 Reconstruction Reward

### 6.0 明确结论：论文的 reward 到底是什么

> **WAM-RL 用来训练 Actor 的 reward，是世界模型预测视频与 Actor 在环境中执行后得到的真实视频之间的相似度。**

论文中的正式定义是：

$$
r_t
=
\operatorname{sim}
\left(
\underbrace{\hat{x}_{t+1:t+H}}_{\text{世界模型预测视频}},
\underbrace{x_{t+1:t+H}}_{\text{环境执行视频}}
\right)
$$

如果使用论文中效果最好的 Pixel MSE，reward 可以直观理解为：

$$
r_t
\approx
-
\frac{1}{H}
\sum_{h=1}^{H}
\left\|
\hat{x}_{t+h}-x_{t+h}
\right\|^2
$$

即：

```text
预测视频和执行视频越相似 → reward 越高
预测视频和执行视频差异越大 → reward 越低
```

这个 reward：

- **用于计算 advantage，并通过 policy gradient 更新 Actor；**
- **不是“任务是否成功”的 reward；**
- **没有直接更新世界模型；**
- **不包含论文明确报告的 success reward 加权项。**

环境提供的 success/fail 信号主要用于筛选成功 rollout，随后使用这些成功视频进行世界模型的 Online Video SFT。论文没有说明将 success/fail 直接加入 Actor reward。

需要注意，论文只写成了 `sim(·,·)`，没有明确给出 Pixel MSE 转换成 reward 时的负号、缩放、归一化和时间聚合方式。因此上面的负 MSE 是依据“误差越大产生越强负信号”的文字所作的合理还原，不是论文给出的完整实现公式。

### 6.1 两段视频分别是什么

在环境时刻 $t$，世界模型先预测未来 $H$ 帧：

$$
\hat{x}_{t+1:t+H}
$$

Actor 随后输出动作并在环境中执行，环境返回真实未来观测：

$$
x_{t+1:t+H}
$$

论文定义：

$$
r_t
=
\operatorname{sim}
\left(
\hat{x}_{t+1:t+H},
x_{t+1:t+H}
\right)
$$

两段视频具体是：

1. 世界模型生成的未来预测视频；
2. Actor 动作在环境中执行后得到的 RGB 观测视频。

论文实验使用 LIBERO 和 RLBench，因此第二段视频实际是仿真器渲染出的 ground-truth RGB 视频，而不是物理机器人摄像头视频。

### 6.2 Pixel MSE

Pixel MSE 直接比较预测帧和环境帧的 RGB 像素：

$$
d_{\text{pixel}}
=
\frac{1}{H}
\sum_{h=1}^{H}
\left\|
\hat{x}_{t+h}-x_{t+h}
\right\|^2
$$

为了让误差越小奖励越高，实际 reward 应类似：

$$
r_t\approx-d_{\text{pixel}}
$$

但论文没有明确给出：

- 负号或从距离到 reward 的转换方式；
- reward 的归一化与裁剪；
- 各帧如何聚合；
- 图像如何归一化；
- 预测帧和环境帧如何进行时间对齐。

### 6.3 Optical Flow MSE

先分别计算两段视频中相邻帧的光流：

$$
\hat{F}_h
=
\operatorname{Flow}
(\hat{x}_{t+h-1},\hat{x}_{t+h})
$$

$$
F_h
=
\operatorname{Flow}
(x_{t+h-1},x_{t+h})
$$

然后比较两个光流场：

$$
d_{\text{flow}}
=
\frac{1}{H-1}
\sum_h
\|\hat{F}_h-F_h\|^2
$$

这里比较的是：

- 预测视频中的二维像素运动；
- 仿真器执行视频中的二维像素运动。

它不是直接比较机器人关节速度，也不是把成功视频和失败视频互相比较。

论文没有说明具体使用了哪一种光流估计算法。

### 6.4 DINOv2 和 V-JEPA2

这两种方法先使用预训练视觉模型提取表示，再比较特征距离：

- DINOv2 主要比较图像级语义表示；
- V-JEPA2 主要比较视频级时空表示。

与 Pixel MSE 相比，它们对局部像素变化不那么敏感，更关注高层语义或动态结构。

### 6.5 论文结果

RLBench Water Plants 上的成功率：

| 方法 | 成功率 |
|---|---:|
| Base | 19% |
| πRL | 18% |
| Pixel MSE | 21% |
| Optical Flow MSE | 19% |
| DINOv2 MSE | 16% |
| V-JEPA2 | 17% |

论文观察到：

- Optical Flow 对成功和失败轨迹的 reward 区分度最强；
- Pixel MSE 的区分度较弱；
- 但 Pixel MSE 的最终任务成功率最高。

作者推测 Pixel MSE：

1. 与世界模型的视频预测训练目标更一致；
2. 对偏离世界模型预测的 OOD 动作提供更强惩罚；
3. 因此可能产生更稳定的策略更新。

这些只是机制解释，论文没有通过额外实验直接证明。

---

## 7. “真实帧”来自哪里

WAM-RL 仍然需要一个可交互环境。

论文实验中的环境是：

- LIBERO 仿真环境；
- RLBench 仿真环境。

仿真器负责：

1. 接收 Actor 输出的动作；
2. 更新机器人和物体状态；
3. 渲染下一时刻 RGB 图像；
4. 返回任务 success/fail 标记。

因此：

```text
世界模型：预测动作执行后可能发生什么
仿真器：真正执行动作并产生观测结果
Reconstruction Reward：比较两者差异
```

世界模型没有替代仿真器。

Figure 1 使用了 “Real World Execution” 的表述，但论文只报告了仿真实验，没有提供真实机器人结果。

理论上，部署到真实机器人时，仿真器可以被真实机器人和摄像头替代；本文没有验证这一点。

---

## 8. 世界模型的 Online Video SFT

### 8.1 使用成功轨迹继续训练

对于最终成功的在线 rollout $x_{1:T}$，世界模型使用视频预测目标继续训练：

$$
L_{\text{video}}
=
\mathbb E_{x_{1:T}}
\left[
\ell(f_\theta(x_{<t}),x_t)
\right]
$$

它被称为 self-supervised，是因为真实未来帧本身就是监督目标，不需要人工逐帧标注。

只使用成功轨迹的目的，是让世界模型继续学习任务相关行为。

成功轨迹也可能包含：

```text
第一次抓取失败
   ↓
调整夹爪位置
   ↓
重新抓取
   ↓
最终成功
```

因此世界模型可能从成功 rollout 中学习失败后的恢复行为。

### 8.2 为什么需要 latent KL

Actor 读取世界模型的中间 latent：

$$
z_t=f_\theta(x_{<t})
$$

如果世界模型在线更新过快，latent 分布会发生变化，Actor 原来学到的“latent 到动作”映射可能失效。

论文保留一个冻结的预训练世界模型：

$$
z_t^{\text{old}}
=
f_{\text{old}}(x_{<t})
$$

由于 DiT latent 是确定性的，作者人为构造高斯近似：

$$
p_\theta(z_t\mid x_{<t})
=
\mathcal N(z_t,\Sigma_\theta)
$$

$$
p_{\text{old}}(z_t\mid x_{<t})
=
\mathcal N(z_t^{\text{old}},\Sigma_{\text{old}})
$$

其中：

- deterministic latent 被视为高斯均值；
- 当前协方差使用 EMA feature statistics 估计；
- 旧模型协方差固定。

KL 正则为：

$$
L_{\text{KL}}
=
\mathbb E_t
\left[
D_{\text{KL}}
\left(
\mathcal N(z_t,\Sigma_\theta)
\|
\mathcal N(z_t^{\text{old}},\Sigma_{\text{old}})
\right)
\right]
$$

最终世界模型损失：

$$
L_{\text{WM}}
=
L_{\text{video}}
+\lambda_{\text{KL}}L_{\text{KL}}
$$

直观上，KL 在限制世界模型“内部语言”变化过快，让 Actor 仍然能理解更新后的 latent。

该设计是工程化近似。论文没有说明：

- KL 施加在哪些 DiT 层；
- token、时间和 batch 维度如何统计；
- EMA 参数；
- $\lambda_{\text{KL}}$ 的具体取值；
- 与 feature L2、cosine loss 等简单约束相比是否更好。

---

## 9. 按正文还原的训练循环

论文没有给出足够完整的算法伪代码。按照正文，可以还原为：

```text
重复：

1. 输入当前观测和任务上下文；
2. 世界模型生成未来预测视频以及中间 latent；
3. Actor 根据 latent，通过 Flow-SDE 采样动作；
4. 在 LIBERO/RLBench 环境中执行动作；
5. 收集环境生成的真实未来视频和 success/fail；
6. 比较预测视频与真实视频，得到 reconstruction reward；
7. 根据 reward 计算 advantage，并通过 policy gradient 更新 Actor；
8. 如果整条 rollout 最终成功：
      使用该视频进行 Online Video SFT；
      同时使用 latent KL 约束世界模型；
9. 使用更新后的两个模块继续交互。
```

论文没有明确给出：

- Actor 和世界模型的更新频率与先后顺序；
- 是否使用 replay buffer；
- 每轮采样多少 rollout；
- reward 的准确尺度；
- advantage 估计方法；
- Flow-SDE 噪声日程及采样步数；
- batch size、optimizer、学习率；
- 总环境交互步数。

因此当前稿件不足以无歧义复现。

---

## 10. RL 是否只是让世界模型预测更准确

不能简单理解成“RL 让世界模型生成图像更准确”。

两条训练路径的作用不同：

### 10.1 Actor RL

```text
世界模型已经生成预测视频
        ↓
调整 Actor 的动作
        ↓
让环境中的真实执行视频接近预测视频
```

RL 直接更新 Actor，不直接更新世界模型。

### 10.2 Online Video SFT

```text
成功的真实执行视频
        ↓
更新世界模型
        ↓
使世界模型更好地建模任务相关过程和恢复行为
```

该部分直接更新世界模型，但属于视频监督/自监督训练，不是策略梯度 RL。

更准确的描述是：

> Actor RL 缩小“想象—执行”的差距；Video SFT 改变被 Actor 执行的想象内容。

此外，论文没有量化证明世界模型的全局视频预测精度显著提高，只定性展示了更新后的模型更容易预测重新抓取等恢复行为。

---

## 11. Reconstruction Reward 与任务成功的关系

### 11.1 核心问题

Reconstruction reward 衡量的是：

> Actor 的真实执行是否符合世界模型的想象。

它没有直接衡量：

> 任务是否完成。

因此：

$$
r_{\text{recon}}\text{ 高}
\quad\not\Rightarrow\quad
\text{任务成功}
$$

例如：

```text
世界模型预测：抓取失败
真实执行结果：抓取失败
两段视频：非常相似
Reconstruction reward：可能很高
任务结果：失败
```

论文在方法部分也明确表示，Actor 被鼓励实现世界模型的预测，而不是直接优化任务特定目标。

### 11.2 任务成功信号如何间接进入

任务成功并非完全消失，而是通过以下路径间接进入：

1. 世界模型预先使用专家轨迹训练，理论上倾向于生成任务相关未来；
2. 世界模型可能受到任务指令或任务上下文影响；
3. 在线训练时，只有最终成功的 rollout 才进入世界模型 Video SFT。

因此任务信号的传播链路是：

```text
success/fail
   ↓
筛选成功视频
   ↓
训练世界模型
   ↓
世界模型产生更面向成功的预测
   ↓
Reconstruction reward 要求 Actor 实现该预测
```

这是一条间接且延迟的任务对齐路径。

### 11.3 “失败但高 reward”的风险

以下情况都可能出现任务失败但 reconstruction reward 较高：

1. 世界模型本身预测了失败或无动作；
2. 抓取成功与失败只占少量像素，静态背景主导 Pixel MSE；
3. 机器人动作错误，但最终画面外观仍与预测接近；
4. 时间对齐误差掩盖了真正的任务差异；
5. 世界模型预测的目标与真实任务目标不一致。

相反，也可能出现任务成功但 reward 较低：

1. Actor 采用了不同但有效的抓取路径；
2. 执行速度与预测视频不同；
3. 物体最终位置正确，但中间轨迹不同；
4. 相机抖动或背景变化导致像素误差较大。

因此 Pixel MSE 会限制策略发现多样化的成功方案。

### 11.4 对探索空间的影响

Flow-SDE 在原始动作空间中加入了随机探索，但 reconstruction reward 会强烈惩罚偏离世界模型预测视频的动作。

所以有效探索目标更接近：

> 在世界模型给出的视觉计划附近，搜索能够实现该计划的动作。

而不是：

> 在整个动作空间中自由发现任意成功策略。

这会降低探索难度，但也会把策略能力限制在世界模型的想象范围内。

---

## 12. 两种不同的反馈信号

需要区分论文中的两个信号：

### 12.1 Reconstruction Reward

- 来源：预测视频和真实执行视频的相似度；
- 主要用途：更新 Actor；
- 特点：稠密，但不直接表示任务成功。

### 12.2 Success/Fail

- 来源：仿真环境的任务完成判定；
- 主要用途：筛选用于世界模型 Online Video SFT 的轨迹；
- 特点：任务相关，但稀疏。

论文没有明确报告将二者组合成 Actor reward：

$$
r_{\text{total}}
=
r_{\text{task}}
+\beta r_{\text{recon}}
$$

因此 Actor 的显式优化目标仍然主要是“实现世界模型的想象”。

---

## 13. 一种更稳妥的奖励设计

更合理的 Actor reward 可以显式结合任务和执行一致性：

$$
r_{\text{total}}
=
\lambda_s r_{\text{success}}
+\lambda_p r_{\text{progress}}
+\lambda_r r_{\text{reconstruction}}
$$

其中：

- $r_{\text{success}}$：任务最终是否成功；
- $r_{\text{progress}}$：物体与目标距离、抓取状态等任务进度；
- $r_{\text{reconstruction}}$：执行是否符合世界模型计划。

三者分别负责：

```text
任务成功奖励：决定“最终要做成什么”
任务进度奖励：提供中间信用分配
重建奖励：约束“怎样稳定地执行计划”
```

这样可以保留 reconstruction reward 的稠密性，同时避免“自洽但失败”的策略获得最高回报。

还可以考虑：

- 使用任务成功判别器或 VLM reward model；
- 对 terminal success 给予明显更高权重；
- 只将 reconstruction reward 作为辅助正则；
- 使用 EMA/frozen target world model，减少 reward 随世界模型更新而漂移；
- 允许与参考视频不同但同样成功的多种轨迹。

---

## 14. 实验结果

### 14.1 主结果

| 方法 | LIBERO-Object | RLBench Water Plants |
|---|---:|---:|
| Base | 68% | 19% |
| πRL（Actor-only） | 78% | 18% |
| WAM-RL | 82% | 22% |

可观察到：

- LIBERO-Object 上，Actor-only 已从 68% 提升到 78%；
- 联合更新进一步达到 82%；
- Water Plants 上，Actor-only 为 18%，没有超过 19% Base；
- WAM-RL 达到 22%。

这些结果在方向上支持作者的观点：

- 短时程任务中，优化 Actor 可以产生明显收益；
- 长时程任务中，更新世界模型可能更重要。

### 14.2 证据强度限制

论文当前不能充分证明“联合更新对长时程任务至关重要”，原因包括：

1. 长时程结论主要来自 Water Plants 一个任务；
2. 绝对增益只有 3 个百分点；
3. 22% 的绝对成功率仍然较低；
4. 没有报告评测回合数；
5. 没有随机种子、标准差或置信区间；
6. 主表 WAM-RL 为 22%，reward 消融 Pixel MSE 为 21%，差异未解释；
7. 没有保持 Actor reward 不变、仅切换 Video SFT 的干净消融；
8. 没有无 KL 的量化对照；
9. 没有真实机器人实验。

因此更保守的结论是：

> 现有实验初步支持联合更新的研究方向，但还不足以证明其稳定性、统计显著性和广泛长时程泛化能力。

---

## 15. 方法的主要优点

### 15.1 关注了 WA 特有的模块耦合

Actor 严重依赖世界模型 latent。在线更新世界模型会造成表示漂移，论文明确识别并处理了这个问题。

### 15.2 在线成功视频包含丰富监督

成功视频不仅包含最终 0/1 结果，还可能包含：

- 接触过程；
- 抓取偏差；
- 失败后的调整；
- 重试和恢复行为。

这些信息可以直接用于改善视频世界模型。

### 15.3 Reconstruction Reward 提供稠密反馈

与只在任务结束时获得一个 success/fail 相比，视频一致性可以提供更加连续的反馈。

### 15.4 揭示了 reward 判别性和可优化性的区别

Optical Flow 更能区分成功与失败，但下游任务表现不如 Pixel MSE，说明 reward 的表示对齐、尺度、方差和优化几何也很重要。

---

## 16. 方法的主要问题

### 16.1 一致性不等于任务成功

这是最重要的问题。Actor 可以忠实实现错误的世界模型预测。

### 16.2 联合训练组件没有被干净隔离

Base、πRL 和 WAM-RL 同时改变了 reward 类型和是否更新世界模型，难以判断收益具体来自哪里。

### 16.3 Reward 随世界模型变化

世界模型持续更新，Actor 的 reward 目标也随之变化。latent KL 不保证输出视频和 reward 数值稳定。

### 16.4 Success-only SFT 存在选择偏差

失败轨迹中也包含有价值的动力学信息，但被完全排除。早期成功率低时，世界模型可能缺少在线训练数据。

### 16.5 Pixel MSE 可能被背景主导

机器人和目标物体通常只占图像的一部分，静态背景可能让成功和失败视频都获得较小误差。

### 16.6 KL 高斯近似较为临时性

使用“deterministic feature 作为均值 + EMA 对角协方差”构造概率分布，是一种工程近似，缺少充分消融。

### 16.7 复现信息不足

关键训练频率、reward 处理、超参数、交互预算等没有完整报告。

### 16.8 实验规模有限

只有 LIBERO-Object 和 RLBench Water Plants，没有多个长时程任务、真实机器人或统计不确定性。

---

## 17. 最值得补充的实验

### 17.1 组件因果拆解

进行 2×2 对照：

```text
Actor reconstruction RL：开 / 关
World Model Video SFT：开 / 关
```

然后单独比较 KL 开/关，并保持 checkpoint、reward、数据和交互预算一致。

### 17.2 多随机种子和置信区间

- 至少运行 3–5 个随机种子；
- 报告评测回合数；
- 给出均值、标准差和 bootstrap 置信区间。

### 17.3 Reward 与任务成功的相关性

直接统计：

- 成功轨迹与失败轨迹的 reward 分布重叠程度；
- `reconstruction reward → success probability` 的校准曲线；
- 高 reward 失败轨迹案例；
- 低 reward 成功轨迹案例。

### 17.4 定量测量恢复行为

报告：

- 失败后重新抓取的触发率；
- 恢复成功率；
- 平均恢复次数；
- 有/无 Video SFT 的差异。

### 17.5 测量世界模型变化

比较更新前后：

- 视频预测误差；
- latent drift；
- 任务相关对象运动误差；
- OOD 动作下的预测误差；
- reward 随训练的漂移。

### 17.6 增加任务奖励

比较：

$$
r_{\text{recon}}
$$

$$
r_{\text{success}}
$$

$$
r_{\text{success}}+\beta r_{\text{recon}}
$$

从而判断 reconstruction reward 更适合作为主目标还是辅助 shaping。

---

## 18. 最终评价

WAM-RL 的核心研究问题是合理的：

> 在 World-Action 模型中，Actor 和世界模型高度耦合，只优化 Actor 可能无法解决长时程预测误差。

论文最有价值的贡献不是某一个具体公式，而是提出了一种协同后训练视角：

- Actor 学习实现世界模型的视觉计划；
- 世界模型从在线成功视频中学习新的任务过程与恢复行为；
- KL 正则维持两者 latent 接口的稳定。

但当前方法将 Actor 的主要 reward 定义为“预测—执行一致性”，没有显式保证任务成功，因此存在：

- 失败但高 reward；
- 成功替代路径被惩罚；
- 策略被限制在世界模型能力范围内；
- reward 与任务目标不一致。

结合有限的实验规模和缺失的统计信息，当前更合适的定位是：

> 一个有启发性的 World-Action 联合在线后训练概念验证，而不是已经充分证明的长时程机器人 RL 方案。

最值得保留的思想是：

> 在线视频不应只被用作 Actor 的 RL 反馈，也可以反过来更新世界模型；但 reconstruction consistency 应更适合作为任务 reward 的辅助项，而不是唯一目标。

