# LaWAM：论文解读与讨论

> 论文：**LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies**  
> 作者：Jialei Chen 等  
> 首次公开时间：2026-06-14  
> 版本与发表平台：arXiv:2606.15768v1，cs.RO，23 页；当前版本为预印本，论文未注明正式会议或期刊  
> 作者单位：清华大学、吉林大学、南开大学、北京大学、哈尔滨工业大学、中关村学院、Striding.AI、无问芯穹（Infinigence AI）  
> 本文档依据本地 [`LaWAM.pdf`](./LaWAM.pdf)、论文正文与附录，以及后续讨论整理。  
> 为避免 Cursor 的公式渲染问题，本文所有公式均使用纯文本代码块。

---

## 1. 一句话概括

LaWAM 不生成未来 RGB 视频，而是预测动作块结束时的 **DINO 未来特征图**，把它作为 latent visual subgoal，再指导动作专家生成真实机器人动作。

核心链路是：

```text
当前图像 + 指令
        ↓
VLM 预测 latent action
        ↓
LaWM 将 latent action 展开成当前场景中的未来特征
        ↓
Action Expert 根据当前特征与未来特征生成动作块
```

论文最有价值的结果不是 LIBERO 上比次优方法高 0.1 个百分点，而是：

- 不需要迭代生成未来视频；
- LaWM 只有约 230M 参数；
- 完整模型约 2.3B 参数；
- 10 个动作去噪步下，每个 action chunk 推理约 187 ms；
- 相对部分 pixel-space WAM，报告最高 24 倍延迟降低。

---

## 2. 研究背景

### 2.1 普通 VLA 的不足

普通 Vision-Language-Action 模型学习：

```text
当前观测 o + 语言指令 l
            ↓
动作块 a1:T
```

概率分解可以写成：

```text
p(a1:T | o, l)
```

这种方法能利用大规模视觉—语言预训练获得较强的语义理解能力，但通常不会显式输出：

```text
“执行这一段动作后，场景会变成什么样”
```

### 2.2 Pixel-space WAM 的不足

World-Action Model 通常先预测未来图像或视频，再根据未来生成动作：

```text
p(a1:T, oT | o, l)
    =
p(oT | o, l)
    ×
p(a1:T | o, oT)
```

其中：

- 第一项预测未来；
- 第二项相当于根据当前状态和期望未来反推动作。

问题是，生成 RGB 视频需要建模大量与控制关系不大的内容：

- 背景纹理；
- 光照；
- 物体表面细节；
- 像素级随机变化；
- 多帧迭代生成过程。

这些内容增加参数量和延迟，但 action chunk 可能只需要知道：

```text
机械臂应该移动到哪里
物体应该发生什么空间变化
动作块结束时场景应达到什么状态
```

### 2.3 LaWAM 的核心判断

作者认为，策略不一定需要可观看的未来视频，只需要一个：

- 紧凑；
- 保留空间结构；
- 包含任务相关变化；
- 能直接被动作生成器使用；

的未来视觉表征。

因此，LaWAM 在冻结视觉编码器的 latent 空间中预测未来：

```text
未来 RGB 视频
    ↓ 替换为
未来 DINO patch 特征图
```

---

## 3. 关键符号与概念

| 符号 | 含义 |
|---|---|
| `o` 或 `o1` | 当前观测图像 |
| `oT` | 经过固定物理时间后得到的 horizon observation |
| `l` | 语言任务指令 |
| `a1:T` | 一个动作块中的连续机器人动作 |
| `f_psi` | 冻结的 DINOv3 视觉编码器 |
| `u` 或 `F1` | 当前图像的 DINO 特征 |
| `uT` 或 `FT` | 真实未来图像的 DINO 特征 |
| `q_phi` | Stage 1 的 IDM / latent-action posterior |
| `z` | IDM 根据当前—未来转移推断的 latent action |
| `z_hat` | Stage 2 中 VLM 根据当前观测和指令预测的 latent action |
| `LaWM_omega` | latent-action-conditioned Latent World Model |
| `uT_tilde` | Stage 1 使用教师 `z` 重建的未来特征 |
| `uT_hat` | Stage 2 使用预测 `z_hat` 得到的未来特征子目标 |
| `p_eta` | Action Expert，生成真实动作块 |
| `tau` | 动作块对应的真实物理时长 |

### 3.1 Latent action 不是真实机器人动作

`z` 不是：

- 关节角；
- 关节速度；
- 夹爪开合；
- 末端执行器位姿；
- 离散动作类别。

它是一个连续 latent code，描述：

```text
“当前视觉特征需要发生什么变化，才能变成未来视觉特征”
```

### 3.2 Latent visual subgoal 不是未来 RGB 图像

`uT_hat` 是稠密的 DINO patch 特征图。它保留一定空间布局，但不能像 RGB 图像那样直接观看。

因此论文使用：

- PCA；
- patch cosine similarity；
- 机械臂区域热图；

来可视化 latent subgoal。

---

## 4. 整体两阶段训练流程

```text
Stage 1：从视觉转移中学习 latent action 和 LaWM

o1, oT
  ↓ 冻结 DINO
u, uT
  ↓ IDM
z
  ↓ 与 u 一起输入 LaWM
uT_tilde
  ↓
匹配真实 uT


Stage 2：教 VLM 在看不到未来时驱动 LaWM

当前图像 + 指令 + 其他视角
            ↓ VLM + LA Query
          z_hat
            ↓
       LaWM(u, z_hat)
            ↓
          uT_hat
            ↓
Action Expert 结合语义上下文生成动作块
```

Stage 1 的作用是定义 latent action 的“语言”；Stage 2 的作用是让策略学会说这种语言。

---

# 5. Stage 1：Latent World Model 训练

## 5.1 Stage 1 的输入

每个训练样本包含：

- 当前观测 `o`；
- 经过固定物理时间 `tau` 后的观测 `oT`；
- 对于有机器人状态的数据，可包含当前末端状态 `s` 和未来末端状态 `sT`。

首先使用冻结的视觉编码器：

```text
u  = f_psi(o)
uT = f_psi(oT)
```

论文采用：

```text
distilled DINOv3 ViT-B/16
```

DINO 在 Stage 1 中冻结，不接受梯度更新。

## 5.2 IDM 的作用

IDM 是 Inverse Dynamics Model。它同时观察当前特征和真实未来特征：

```text
z ~ q_phi(z | u, uT)
```

它的任务不是预测真实机器人控制指令，而是提取一个能够解释这次视觉转移的 latent action。

直观上：

```text
输入：变化前 + 变化后
输出：发生了什么变化
```

IDM 使用 24 层 Transformer，采用类似 V-JEPA2 的时空设计：

1. 展平当前与 horizon observation 的视觉 patch；
2. 拼成一个 token 序列；
3. 联合处理两个时刻；
4. 输出连续 latent-action posterior。

论文对 `q_phi` 使用相对标准的 VAE 式表达：

```text
q_phi(z | u, uT)
```

并通过 KL 项将其正则到标准正态分布附近。

需要注意：论文没有清楚报告 latent action 的 token 数、具体维度和 posterior 参数化细节。这是重要复现缺口。

## 5.3 LaWM 的作用

LaWM 接收：

- 当前视觉特征 `u`；
- IDM 推断的 latent action `z`。

然后预测 horizon feature：

```text
uT_tilde = LaWM_omega(u, z)
```

LaWM 也是 24 层 Transformer，约 230M 参数。

它使用 adaptive layer normalization 将 `z` 注入 decoder，而不是简单把 latent action 加到视觉 token 上。

作者给出的理由是：

- additive injection 容易受到 latent-action norm 波动影响；
- 可能使所有视觉 token 发生全局漂移；
- 在跨具身训练中会出现明显 loss spike；
- AdaLN 更稳定。

## 5.4 Stage 1 完整损失

论文的 Stage 1 目标为：

```text
L_LAM =
    L_wm
  + L_aux
  + beta * L_KL
```

其中：

```text
L_wm  = ||uT_tilde - uT||^2

L_aux = ||g(s, z) - sT||^2

L_KL  = KL(
          q_phi(z | u, uT)
          ||
          Normal(0, I)
        )

beta = 1e-5
```

下面分别解释每一项。

### 5.4.1 `L_wm`：未来视觉特征预测损失

```text
L_wm = ||uT_tilde - uT||^2
```

其中：

- `uT` 是真实 horizon observation 的 DINO 特征；
- `uT_tilde` 是 LaWM 根据 `u + z` 预测的特征。

它训练：

- IDM 产生对 decoder 有用的 `z`；
- LaWM 根据当前状态和 `z` 预测未来；
- 二者共同形成 latent transition autoencoder。

如果只有这一项，IDM 可能把大量未来信息直接塞入 `z`，因此论文又加入辅助状态预测与 KL 正则。

### 5.4.2 `L_aux`：末端状态预测损失

```text
L_aux = ||g(s, z) - sT||^2
```

其中：

- `s` 是当前机器人末端执行器状态；
- `sT` 是 horizon 时刻的末端状态；
- `g` 是轻量 MLP predictor；
- `z` 是 IDM 推断的 latent action。

它迫使 `z` 包含具身运动信息，而不只是：

- 光照变化；
- 背景变化；
- 外观变化；
- 视觉增广差异。

因为不同机器人状态语义不完全一致，论文使用轻量的、按具身处理的 MLP heads。

Stage 1 结束后，这个辅助预测器会被丢弃。

论文没有明确说明没有本体状态的人类视频如何应用 `L_aux`。合理实现应只对具备相应状态标签的机器人样本启用该项，但当前 arXiv v1 没有给出完整 mask 或数据分支定义。

### 5.4.3 `L_KL`：latent-action 分布正则

```text
L_KL =
  KL(
    q_phi(z | u, uT)
    ||
    Normal(0, I)
  )
```

作用包括：

1. 限制 `z` 的信息容量；
2. 让 latent-action 空间更平滑；
3. 防止 IDM 为每个训练样本创建完全离散、不可预测的代码；
4. 让 Stage 2 的 policy prior 更容易拟合 `z`；
5. 提高不同数据源和具身之间的 latent 可比性。

`beta = 1e-5` 很小，因此该项更像轻量正则，而不是强信息瓶颈。

### 5.4.4 Stage 1 的梯度流

```text
L_wm
  → 更新 IDM
  → 更新 LaWM

L_aux
  → 更新 IDM
  → 更新辅助状态 predictor

L_KL
  → 更新 IDM posterior

DINO
  → 明确冻结
```

## 5.5 Stage 1 为什么不需要真实动作标签

核心原因是 IDM 从视觉结果中反推动作代码：

```text
(u, uT) → z
```

训练只需要知道：

```text
画面之前是什么样
经过一段时间后是什么样
```

因此，Stage 1 可以利用：

- 无机器人动作标签的视频；
- 没有语言指令的视频；
- 第一人称人类操作视频；
- 不同具身、不同控制空间的视频。

人类视频贡献的是视觉动力学先验，而不是直接教机器人输出人类动作。

## 5.6 Stage 1 的训练数据与超参数

| 项目 | 设置 |
|---|---|
| 机器人视频 | 约 3,000 小时 |
| 第一人称人类视频 | 约 1,500 小时 |
| 总视频量 | 约 4,500 小时 |
| GPU | 16×H100 |
| 训练步数 | 100k |
| 全局 batch size | 1024 |
| 优化器 | AdamW |
| 学习率 | 3e-4 |
| weight decay | 1e-2 |
| KL 权重 `beta` | 1e-5 |
| 机器人视频 horizon | 1.2 s |
| 人类视频 horizon | 0.4 s |

在进入 DINO 前，encoder 和 decoder view 使用不同随机裁剪和颜色增强，但同一 clip 内的增强在时间上保持一致。

目的在于：

- 减少对固定像素布局的记忆；
- 鼓励跨视角、跨具身的转移编码；
- 避免 IDM 只利用低级像素对应。

## 5.7 Stage 1 结束后保留什么

```text
保留：
  LaWM decoder
  IDM posterior encoder（Stage 2 产生 teacher z）

丢弃：
  辅助末端状态 predictor

部署时不使用：
  IDM
  真实未来观测
```

---

# 6. Stage 2：Latent World Action Model 训练

## 6.1 为什么需要 Stage 2

Stage 1 的 IDM 需要真实未来：

```text
z = IDM(u, uT)
```

但部署时 `uT` 还没有发生，因此无法调用 IDM。

Stage 2 训练一个 policy prior：

```text
z_hat = p_theta(z | o, l)
```

让策略只根据：

- 当前图像；
- 语言指令；
- 可选辅助视角；

预测能够驱动 LaWM 的 latent action。

## 6.2 VLM 与 LA Query

LaWAM 使用：

- Qwen3-VL-2B 的前 16 层；
- hidden size 1024；
- latent-action query tokens；
- action query tokens；
- 主视角、语言指令和辅助视角作为上下文。

输入顺序大致为：

```text
主视角
语言指令
LA Query
其他视角
Action Query
```

通过 causal attention mask：

- LA Query 聚合预测 `z_hat` 所需的信息；
- Action Query 仍然可以读取完整语义上下文；
- 轻量 query-aggregation block 将 LA Query 映射为与 Stage 1 `z` 相同形状的 `z_hat`。

## 6.3 IDM teacher 与 VLM student

Stage 2 训练样本仍有真实未来，因此可以使用 Stage 1 IDM 得到教师 latent action：

```text
z_teacher = IDM(u, uT)
```

VLM 根据当前信息预测：

```text
z_hat = PolicyPrior(o, l)
```

二者直接进行 latent-action distillation。

这类似 conditional VAE 中：

```text
posterior：看见目标后编码 z
prior：不看目标，预测 z
decoder：使用 z 预测目标
```

对应到 LaWAM：

```text
IDM posterior：
  q_phi(z | u, uT)

Policy prior：
  p_theta(z_hat | o, l)

Shared decoder：
  LaWM(u, z)
```

## 6.4 LaWM 产生 policy-driven subgoal

Stage 2 中，LaWM 实际接收的是 VLM 输出的 `z_hat`：

```text
uT_hat = LaWM_omega(u, z_hat)
```

因此并不是：

```text
Stage 1 训练 LaWM
推理时突然把 IDM 换成 VLM
```

而是 Stage 2 已经把 VLM 输出放入 LaWM 计算图中，专门训练二者接口。

## 6.5 Action Expert

Action Expert 使用 4 个 Alternate-DiT blocks，对应总计 16 层 Transformer。

它接收两个信息流：

### 语义流

```text
VLM hidden states
```

主要描述：

- 指令要求什么；
- 操作哪个物体；
- 当前场景中的语义关系；
- 多视角上下文。

### 动力学流

```text
(u, uT_hat)
```

主要描述：

- 当前场景是什么样；
- action chunk 结束时应变成什么样；
- 当前到子目标之间的空间变化。

Alternate-DiT 在动作去噪过程中交替使用：

```text
Self Attention
→ Inverse-Dynamics Attention
→ Self Attention
→ Semantic Attention
```

直观上：

- Self Attention：协调动作块内部不同时间步；
- Inverse-Dynamics Attention：根据当前—未来特征推断需要的运动；
- Semantic Attention：重新读取任务意图与视觉语义。

## 6.6 Stage 2 完整损失

论文定义：

```text
L_LaWAM =
    lambda_distill * L_distill
  + lambda_wm      * L_subgoal
  + L_act
```

其中：

```text
L_distill =
  E[ ||z_hat - z_teacher||^2 ]

L_subgoal =
  ||uT_hat - uT||^2

L_act =
  conditional flow-matching loss

lambda_distill = 0.1
lambda_wm      = 0.1
```

论文在公式中继续把 `L_subgoal` 写作 `L_wm`。为了避免和 Stage 1 使用 teacher `z` 的 `L_wm` 混淆，本文称其为 `L_subgoal`。

### 6.6.1 `L_distill`：latent-action 蒸馏

```text
L_distill =
  E[ ||z_hat - z_teacher||^2 ]
```

其中：

- `z_teacher` 来自看过真实未来的 IDM；
- `z_hat` 来自只看当前观测和指令的 VLM。

作用：

1. 让 VLM 输出与 IDM latent 具有相同形状；
2. 直接固定 latent 坐标系；
3. 防止 VLM 与 LaWM 通过任意旋转或重参数化自行形成另一套接口；
4. 让 `z_hat` 落入 LaWM 在 Stage 1 见过的 latent 区域。

它保证的是近似对齐，不可能保证完全相同，因为 IDM 拥有未来信息，VLM 没有。

### 6.6.2 `L_subgoal`：policy-driven 未来特征监督

```text
uT_hat = LaWM(u, z_hat)

L_subgoal =
  ||uT_hat - uT||^2
```

这是一层功能对齐：

```text
不仅要求 z_hat 数值接近 z_teacher，
还要求它经过 LaWM 后能产生正确未来特征。
```

即使存在多个 latent code 能产生相似未来，只要：

```text
LaWM(u, z_hat) ≈ uT
```

该 `z_hat` 对策略仍然是有效的。

### 6.6.3 `L_act`：条件 Flow Matching 动作损失

论文只说明 `L_act` 是条件 flow-matching loss，没有在正文中重新给出完整插值公式。

一种标准表达是：

```text
epsilon ~ Normal(0, I)
t       ~ Uniform(0, 1)

a_t =
  (1 - t) * epsilon
  + t * a_data

v_target =
  a_data - epsilon

L_act =
  E[
    ||v_eta(a_t, t | o, l, u, uT_hat)
      - v_target||^2
  ]
```

其中：

- `a_data` 是示范中的真实动作块；
- `epsilon` 是高斯噪声动作；
- `a_t` 是噪声与真实动作之间的插值；
- Action Expert 学习将噪声运输到真实动作分布的速度场。

不同实现可能反转时间方向，因此 target velocity 的符号也可能反转。LaWAM 当前稿件没有给出这一实现约定。

推理时从噪声开始：

```text
Gaussian action noise
        ↓ 10 个 flow integration / denoising steps
continuous EEF action chunk
```

## 6.7 三个损失分别解决什么问题

| 损失 | 监督对象 | 主要作用 |
|---|---|---|
| `L_distill` | `z_hat` 对齐 `z_teacher` | 对齐 VLM 与 IDM 的 latent 坐标 |
| `L_subgoal` | `LaWM(u, z_hat)` 对齐 `uT` | 保证预测 latent 能正确驱动 LaWM |
| `L_act` | 预测 flow 对齐真实动作 flow | 训练 Action Expert 输出可执行动作块 |

可以概括为：

```text
L_distill：VLM 要说对 latent 语言

L_subgoal：说出的 latent 要能预测对未来

L_act：动作专家要能实现这个未来
```

## 6.8 Knowledge Insulation 与 LaWM 是否冻结

论文明确写道：

```text
使用 Knowledge Insulation，
防止 Action Expert 的梯度覆盖预训练 LaWM 动力学。
```

因此可以确定：

```text
L_act 不应反向破坏 LaWM
```

更直观的梯度图是：

```text
L_distill
  → Policy Prior / VLM

L_subgoal
  → Policy Prior / VLM
  → 可能更新 LaWM

L_act
  → Action Expert
  × 通过 KI 阻断对 LaWM 动力学的破坏
```

### 论文没有明确回答的地方

当前 arXiv v1 没有明确写出：

```text
Stage 2 中 LaWM 是否对所有损失 requires_grad = false
```

存在两种实现可能。

#### 方案 A：LaWM 完全冻结

```text
固定 LaWM 参数
L_subgoal 经过固定 LaWM 的 Jacobian
只更新产生 z_hat 的 Policy Prior
```

这样 LaWM 相当于固定的、可微的 latent decoder。

#### 方案 B：LaWM 只与动作损失隔离

```text
L_subgoal 可以小学习率微调 LaWM
L_act 通过 Knowledge Insulation 不更新 LaWM
```

正文使用“pretrained LaWM”而不是明确的“frozen LaWM”，附录又写：

```text
Action Expert learning rate = 1e-4
all other modules learning rate = 3e-5
```

这些信息更倾向于方案 B，但不能构成决定性证明，因为“all other modules”也可能只指其他可训练模块。

因此最严谨的结论是：

```text
确定：
  Action Expert 的 L_act 不更新 LaWM

很可能：
  L_subgoal 可以小学习率更新 LaWM

不确定：
  作者实现是否将 LaWM 参数在 Stage 2 完全冻结
```

要完全确认，需要作者代码、optimizer 参数组或明确的 `requires_grad` 配置。

## 6.9 Stage 2 训练数据与超参数

Stage 2 只使用带语言指令的机器人轨迹。

人类第一人称视频通常没有明确描述机器人目标的语言标签，因此：

```text
人类视频：
  只通过 Stage 1 影响 LaWM dynamics prior

机器人语言轨迹：
  用于 Stage 2 policy integration
```

训练设置：

| 项目 | 设置 |
|---|---|
| Policy integration 训练步数 | 200k |
| GPU | 64×H100 |
| 全局 batch size | 1024 |
| Action Expert 学习率 | 1e-4 |
| 其他模块学习率 | 3e-5 |
| `lambda_distill` | 0.1 |
| `lambda_wm` | 0.1 |
| 默认推理去噪步数 | 10 |

---

# 7. 混合控制频率与物理时间编码

## 7.1 问题

不同机器人数据集可能运行在：

- 5 Hz；
- 10 Hz；
- 20 Hz；
- 其他原生控制频率。

相同动作 token 下标不代表相同真实时间：

```text
5 Hz 的第 5 个 token  = 1.0 s
20 Hz 的第 5 个 token = 0.25 s
```

如果直接混合训练，模型会混淆：

```text
“第 i 个动作 token 对应动作执行到什么阶段”
```

## 7.2 固定物理 horizon

对控制频率为 `h_b` 的分支 `b`，动作块长度为：

```text
H_b = round(tau * h_b)
```

例如固定：

```text
tau = 1.0 s
```

则：

```text
5 Hz  → 5 个 action tokens
10 Hz → 10 个 action tokens
20 Hz → 20 个 action tokens
```

这些动作块都对应相同的真实时间，并共同指向 `tau` 时刻的 LaWM subgoal。

## 7.3 物理时间位置编码

分支 `b` 中第 `i` 个动作 token 的真实时间为：

```text
t_b_i = i / h_b
```

然后使用 sinusoidal encoding：

```text
phi(t_b_i) =
  Concat[
    sin(t_b_i * omega_k),
    cos(t_b_i * omega_k)
  ]
```

该编码加到 action-query embedding 上。

因此：

```text
相同真实时间、不同 token 下标
  → 得到相同时间含义

相同 token 下标、不同控制频率
  → 得到不同时间编码
```

## 7.4 Stage 1 中仍存在的时间尺度疑问

论文对：

- 机器人视频使用 1.2 s horizon；
- 人类视频使用 0.4 s horizon。

物理时间编码解决的是 Stage 2 action token 的频率歧义，但论文没有充分说明：

```text
不同 horizon 下学习到的 Stage 1 latent action
如何进入统一 latent 分布
```

这可能使 `z` 同时编码：

- 运动类型；
- 运动幅度；
- 时间尺度。

---

# 8. 推理流程

部署时不存在：

- 真实未来 `oT`；
- 真实未来特征 `uT`；
- IDM teacher；
- 蒸馏损失；
- subgoal supervision。

完整推理链为：

```text
1. 当前主视角经过 DINO：

   u = DINO(o)

2. VLM 根据当前图像与指令预测 latent action：

   z_hat = PolicyPrior(o, l)

3. LaWM 单次前向预测 latent visual subgoal：

   uT_hat = LaWM(u, z_hat)

4. Action Expert 从噪声生成动作块：

   a1:T =
     ActionExpert(
       VLM_context,
       u,
       uT_hat,
       noise
     )

5. 执行动作块，获取新观测

6. 使用新观测重复上述流程
```

需要特别区分：

```text
LaWM future prediction：
  单次前向，非迭代

Action Expert：
  仍然使用约 10 个 flow denoising steps
```

论文报告的 187 ms 是完整 action-chunk 预测延迟，而不是只计算 LaWM 的时间。

---

# 9. 对关键问题的讨论

## 9.1 IDM 为什么看起来很强

因为 IDM 在训练时看到了结果：

```text
输入：
  当前特征 u
  真实未来特征 uT

输出：
  总结这次转移的 z
```

它不是在不知道未来时进行规划，而是对已经发生的变化做编码。

真正困难的是 Stage 2：

```text
VLM 只看当前和指令
却要预测 IDM 看过未来后才得到的 z
```

## 9.2 Latent action 是谁 embedding 的

存在两层编码：

```text
DINO：
  图像 → 视觉特征

IDM：
  当前—未来视觉转移 → latent action z
```

没有一个真实动作先被单独 embedding 成 `z`。

Stage 2 中：

```text
VLM hidden states
  ↓ LA Query
  ↓ query aggregation / projection
z_hat
```

因此：

- Stage 1 的 `z` 由 IDM 定义；
- Stage 2 的 `z_hat` 由 VLM 预测；
- 两者通过直接 L2 蒸馏和 LaWM subgoal loss 对齐。

## 9.3 IDM 的 `z` 和 VLM 的 `z_hat` 能一样吗

分三个层面：

```text
张量形状：
  一样，由网络接口保证

latent 坐标系：
  一样，由 L_distill 直接固定

具体数值：
  通常不可能完全一样
```

原因是：

```text
IDM 输入：
  u + 真实 uT

VLM 输入：
  当前图像 + 指令
```

最重要的功能条件是：

```text
LaWM(u, z_hat)
  ≈
LaWM(u, z_teacher)
  ≈
uT
```

## 9.4 Posterior-prior gap

同一个当前状态与指令可能有多条有效未来：

```text
从左侧抓杯子
从右侧抓杯子
先调整姿态再抓
直接接近后抓
```

IDM 看到了数据中真实发生的轨迹，因此可以输出对应的 `z_teacher`。

VLM 没有未来信息，只能根据当前和指令预测：

```text
p_theta(z | o, l)
```

当未来是多模态时，L2 蒸馏可能使 `z_hat` 接近多个 teacher latent 的平均值。

论文的缓解方式包括：

- Stage 1 KL 正则；
- `L_distill`；
- `L_subgoal`；
- Action Expert 在训练时使用 policy-driven subgoal；
- 每个 action chunk 后重新观察和预测。

但论文没有使用：

- 多 latent 采样；
- best-of-N；
- 显式不确定性；
- 候选未来打分；
- MPC。

## 9.5 Latent action 会不会作弊

因为 IDM 看到了 `uT`，它可能直接把大量未来信息编码进 `z`，而不是提取可泛化动力学。

论文用于限制这一问题的方法：

- latent bottleneck；
- KL 正则；
- 辅助末端状态预测；
- 不同图像增强；
- LaWM 必须结合当前 `u` 才能预测 `uT`。

但是：

- `beta = 1e-5` 较小；
- latent action 维度没有明确报告；
- 没有 latent capacity sweep；
- 没有 mutual information 或 KL utilization 分析；
- 没有充分排除 `z` 是高带宽 future code。

## 9.6 为什么不直接把 latent action 给 Action Expert

抽象 `z` 主要描述“发生什么变化”，但缺少当前场景的具体空间信息。

例如同一个抽象转移：

```text
“机械臂移动到把手并拉开”
```

在不同场景中对应：

- 不同把手位置；
- 不同机器人外形；
- 不同相机视角；
- 不同机械臂当前位置；
- 不同抽屉方向。

LaWM 执行：

```text
(当前场景 u, 抽象变化 z)
              ↓
场景化未来特征 uT_hat
```

这使 Action Expert 获得空间化、具身绑定的目标。

## 9.7 LaWAM 是不是经典世界模型规划

不是严格意义上的经典 planning / MPC。

LaWAM 推理时：

- 不枚举多个候选真实动作；
- 不预测每个候选动作的后果；
- 不比较多条未来轨迹；
- 不使用任务价值函数选择 rollout；
- 只由 policy prior 预测一个 `z_hat`；
- 只生成一个 chunk-level subgoal。

因此更准确的说法是：

```text
显式未来特征条件化的 VLA
```

而不是已经具备成熟反事实规划能力的 world model。

---

# 10. 实验

## 10.1 实验覆盖范围

论文从五个方向评测：

1. 模拟 benchmark 性能；
2. 推理效率；
3. 真实机器人迁移；
4. latent dynamics 行为；
5. 组件消融。

主要 benchmark：

- LIBERO；
- RoboTwin 2.0；
- 真实 Pick-and-Place；
- 真实 Drawer Opening；
- 真实 Towel Folding。

## 10.2 LIBERO

### 训练与评测协议

- 四个标准 suite：
  - LIBERO-Spatial；
  - LIBERO-Object；
  - LIBERO-Goal；
  - LIBERO-Long。
- 移除失败 demonstrations；
- 后训练 25k steps；
- global batch size 256；
- 40 个任务；
- 每任务 50 次；
- 总计 2,000 次 trial。

### 主结果

| 方法 | 类型 | 模型大小 | 延迟 ms | Long | Goal | Object | Spatial | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenVLA-OFT | VLA | 7B | — | 94.5 | 97.9 | 98.4 | 97.6 | 97.1 |
| pi0 | VLA | 3.5B | 220 | 88.4 | 94.4 | 96.8 | 98.0 | 94.4 |
| pi0.5 | VLA | 3.5B | 220 | 92.4 | 98.0 | 98.2 | 98.8 | 96.9 |
| GR00T-N1.6 | VLA | 3.3B | 259 | 94.4 | 97.5 | 98.5 | 97.7 | 97.0 |
| LAPA | Latent action | 7B | — | 55.4 | 58.8 | 74.6 | 73.8 | 65.7 |
| UniVLA | Latent action | 7B | — | 92.0 | 95.6 | 96.8 | 96.5 | 95.2 |
| Mantis | Latent action | 5.8B | — | 94.2 | 94.4 | 99.2 | 98.8 | 96.7 |
| VLA-JEPA | Latent action | 3B | — | 95.8 | 97.2 | 99.6 | 96.2 | 97.2 |
| F1 | Pixel WAM | 4B | 399 | 91.3 | 95.4 | 97.8 | 98.2 | 95.7 |
| Motus | Pixel WAM | 8B | 3231 | 97.6 | 96.6 | 99.8 | 96.8 | 97.7 |
| Cosmos-Policy | Pixel WAM | 2.1B | 1413 | 97.6 | 98.2 | 100.0 | 98.1 | 98.5 |
| LingBot-VA | Pixel WAM | 5.5B | 4482 | 98.5 | 97.2 | 99.6 | 98.5 | 98.5 |
| Fast-WAM | Pixel WAM | 6B | 486 | 95.2 | 97.0 | 100.0 | 98.2 | 97.6 |
| **LaWAM** | **Latent WAM** | **2.3B** | **187** | **97.0** | **98.4** | **99.6** | **99.4** | **98.6** |

### 如何解读

正面结论：

- LaWAM 在表中平均成功率最高；
- 比 pi0.5 快约 1.18 倍；
- 比 Fast-WAM 快约 2.6 倍；
- 比 Cosmos-Policy 快约 7.6 倍；
- 比 LingBot-VA 快约 24.0 倍；
- 在不生成未来 RGB 视频的情况下保持高成功率。

需要保守的地方：

```text
LaWAM：98.6%
次优：98.5%
差距：0.1 percentage point
```

总计 2,000 次 trial 时，0.1 个百分点约等于 2 次成功的数量级。不同模型并非配对同一随机运行，且论文未报告：

- 多训练种子；
- 标准差；
- bootstrap confidence interval；
- 显著性检验。

因此：

```text
可以说 LaWAM 达到 SOTA/competitive 水平
不宜强调其精度显著超过所有方法
```

## 10.3 RoboTwin 2.0

### 训练与评测协议

- 50 个双臂任务；
- 2,500 条 clean-scene demonstrations；
- 25,000 条 heavy-randomization demonstrations；
- 后训练 100k steps；
- global batch size 1024；
- 64×H100；
- 约 20 小时；
- 每任务 clean 100 次；
- 每任务 randomized 100 次。

### 平均结果

| 方法 | Clean SR | Randomized SR | 两者平均 |
|---|---:|---:|---:|
| Fast-WAM | 91.98 | 90.52 | 91.25 |
| GigaWorld-Policy | 86.36 | 85.04 | 85.70 |
| LingBot-VA | 91.50 | **90.92** | 91.21 |
| pi0.5 | 82.74 | 76.76 | 79.75 |
| Motus | 88.66 | 87.02 | 87.84 |
| **LaWAM** | **92.64** | 89.80 | **91.22** |

### 如何解读

- LaWAM 在 clean setting 第一；
- clean 比 Fast-WAM 高 0.66 个百分点；
- randomized setting 不是第一；
- randomized 比 LingBot-VA 低 1.12 个百分点；
- randomized 比 Fast-WAM 低 0.72 个百分点；
- clean/random 平均约 91.22%，略高于 Fast-WAM 与 LingBot-VA。

论文正文说 LaWAM 在随机场景中“接近最强 pixel-space WAM”，这一措辞是准确的。

### LaWAM 的明显弱项

从附录 Table 4 看，部分任务仍明显困难：

| 任务 | Clean | Randomized |
|---|---:|---:|
| Hanging Mug | 51 | 43 |
| Open Microwave | 41 | 43 |
| Turn Switch | 47 | 56 |
| Place Can Basket | 92 | 65 |
| Stack Blocks Three | 90 | 75 |
| Stack Bowls Three | 90 | 80 |

这些任务涉及：

- 精细接触；
- 几何约束；
- 铰接物体；
- 多阶段堆叠；
- 对随机位置和姿态敏感的操作。

## 10.4 真实机器人实验

### 结果

每个任务 30 次：

| 方法 | Pick-and-Place | Open Drawer | Fold Towel | Average |
|---|---:|---:|---:|---:|
| pi0.5 | 86.7 | 80.0 | 83.3 | 83.3 |
| GR00T-N1.6 | 83.3 | 76.7 | 46.7 | 68.9 |
| Fast-WAM | 56.7 | 63.3 | 70.0 | 63.3 |
| LingBot-VA | 76.7 | 83.3 | 0.0 | 53.3 |
| **LaWAM** | **93.3** | **86.7** | **90.0** | **90.0** |

### 平台与数据

#### Pick-and-Place

- Franka Emika Panda；
- 平行夹爪；
- 外部 RGB 相机；
- 150 条真实 demonstrations；
- 目标物包括毛绒蔬菜和塑料鸭；
- 包含部分训练空间外的位置。

#### Drawer Opening

- Franka Emika Panda；
- 外部 RGB 相机；
- 150 条真实 demonstrations；
- 抽屉位置随机；
- 朝向扰动最高约 15 度；
- 抽拉方向不准确会产生明显阻力。

#### Towel Folding

- Quanta X1 双臂机器人；
- 280 条 demonstrations；
- 先抖开并铺平毛巾；
- 两次长边折叠；
- 一次短边折叠；
- 熟练遥操作员约需 70 s；
- 强调长时程、双臂协调和动态布料交互。

### 如何解读

LaWAM 在三项任务中都排名第一，尤其毛巾折叠达到 90%。

论文认为高延迟方法在毛巾仍继续运动时会产生 stale observation：

```text
观测时的毛巾状态
   ↓ 模型推理等待
动作输出时毛巾已经变化
   ↓
动作与当前状态不再匹配
```

这使低延迟不仅是吞吐指标，也直接影响动态任务成功率。

统计上仍需谨慎：

- 每任务只有 30 次；
- 成功率最小分辨率为 3.33 个百分点；
- LaWAM 90.0% 对 pi0.5 83.3%，总计约多成功 6/90 次；
- 没有置信区间或多次独立训练。

## 10.5 LaWM dynamics 分析

### 跨环境、跨具身 open-loop rollout

作者先从源视频中提取 latent-action sequence，然后：

```text
同一 latent-action sequence
        +
不同环境或具身的初始特征
        ↓
LaWM 迭代 rollout
```

观察到不同初始场景产生上下文相关的 latent 变化。

作者据此解释：

```text
latent action：
  表示相对抽象、跨具身的视觉转移

LaWM：
  使用当前视觉状态将转移绑定到具体具身与场景
```

但这些结果主要是 latent 可视化，不是未见具身上的真实闭环控制成功率。

### 500 条 LIBERO rollout 分析

Appendix Figure 10 比较：

1. rollout feature 与 ground-truth future feature；
2. ground-truth future 与 initial feature；
3. rollout feature 与 initial feature。

结果显示：

- rollout 与真实未来保持较高相似度；
- rollout 与初始状态的相似度下降；
- 模型不是简单复制初始特征。

这支持 LaWM 学到非平凡 latent dynamics，但仍未测试：

```text
给定多个真实候选动作时，
LaWM 能否准确区分其不同未来后果
```

## 10.6 组件消融

Figure 6 比较：

- `w/o WM`；
- `w/o KI & distill`；
- `w/o distill`；
- `w/o pretrain`；
- 完整 LaWAM。

论文文字结论：

1. 移除 LaWM 降幅最大；
2. LIBERO-Long 受影响尤其明显；
3. 移除 latent-action distillation 会明显下降；
4. 同时移除 KI 与 distillation 进一步下降；
5. LaWM 需要既被对齐的 latent action 驱动，又不被 Action Expert 梯度破坏。

需要注意：

- 正文只给图，没有完整数值表；
- `w/o WM` 是否做参数量匹配没有充分说明；
- 因此不能完全排除增益来自额外容量和辅助监督；
- 缺少 shuffled subgoal、random subgoal、ground-truth subgoal upper bound。

## 10.7 混频训练消融

Appendix Figure 7 使用同一批原生 20 Hz LIBERO 轨迹，下采样得到：

- 10 Hz；
- 5 Hz。

然后联合训练 5/10/20 Hz 数据。

比较：

- `w/o pos`；
- `w/ pos`；
- `only 20Hz`。

结果方向：

```text
混频但无物理时间编码
  → 明显退化

加入物理时间编码
  → 大部分恢复到原生 20 Hz 参考附近
```

这个实验较好地控制了：

- 任务分布；
- 具身；
- 视觉域；
- 语言指令；
- 专家轨迹数量。

因此，它是论文中因果隔离相对干净的一项实验。

## 10.8 推理效率

论文测量方式：

- A100 GPU；
- 10 个 action denoising steps；
- 重复 1,000 次 action-chunk prediction；
- 报告平均模型墙钟时间；
- 不含 simulator 和真实机器人执行。

LaWAM：

```text
187 ms / action chunk
```

世界建模部分：

```text
LaWM：约 230M
典型 WAN video backbone：约 5B
```

按这两个数字计算，world-modeling 参数减少约 95%。

但参数比较需要注意：

- 论文对 WAM 参数计数排除了 video-diffusion VAE；
- 也排除了可达 10B 的 text encoder；
- baseline 延迟部分来自原论文、部分来自作者复现；
- 软件栈、精度格式、缓存、多视角数和采样器可能不完全一致。

因此：

- 24 倍数量级差距的方向可信；
- 较小的 1.2–2.6 倍差距需要统一实现进一步验证。

---

# 11. 附录解读

## 11.1 Appendix A：RoboTwin 逐任务结果

Appendix Table 4 给出了 50 个任务的 clean/randomized 成功率。

主要作用：

- 说明平均值不是由少量简单任务决定；
- 暴露 LaWAM 在部分精细接触与随机化任务上的弱点；
- 展示不同 WAM 方法的任务偏好差异。

LaWAM 在很多任务达到或接近 100%，例如：

- Adjust Bottle；
- Click Alarmclock；
- Click Bell；
- Grab Roller；
- Lift Pot；
- Open Laptop；
- Place Container Plate；
- Shake Bottle；
- Shake Bottle Horizontally。

但在 Hanging Mug、Open Microwave、Turn Switch 等任务明显较低，说明 latent subgoal 并没有消除精细几何与接触控制难题。

## 11.2 Appendix B：定性可视化协议

LaWM 不重建 RGB，因此作者使用如下方式画 subgoal heatmap：

```text
1. 在初始观测中选择一个机械臂 patch

2. 提取该 patch 的 DINO feature

3. 计算它与 predicted subgoal 中每个 patch 的 cosine similarity

4. 将 similarity 画成热图
```

如果初始机械臂 patch 在 subgoal 中的高相似度区域发生移动，可以解释为：

```text
LaWM 预测机械臂特征将移动到该区域
```

论文观察到 action chunk 执行过程中，真实机械臂逐渐接近热图预测区域。

局限：

- 热图以人为选择的机械臂 patch 为中心；
- 主要说明空间对应；
- 不是完整未来场景重建；
- 不能直接证明物体接触、力和细粒度形变预测正确；
- PCA 颜色没有固定物理语义。

## 11.3 Appendix C.1：LaWM 架构

### 视觉编码器

```text
distilled DINOv3 ViT-B/16
```

### IDM encoder

- 24 层 Transformer；
- V-JEPA2 风格时空设计；
- 当前与 horizon patch 展平后联合处理；
- 输出 continuous latent-action posterior。

### LaWM decoder

- 24 层 Transformer；
- 约 230M 参数；
- 使用 AdaLN 条件化 latent action；
- 输出未来 DINO feature map。

### 辅助状态 predictor

- 轻量 MLP；
- 处理不同机器人状态语义；
- Stage 1 后丢弃。

## 11.4 Appendix C.2：LaWAM 架构

- Qwen3-VL 前 16 层；
- 4 个 Alternate-DiT blocks；
- Action Expert 合计 16 个 Transformer 层；
- hidden dimension 1024；
- 输入 256×256 RGB；
- 多视角可选；
- 不输入 proprioceptive state；
- 动作统一转换为 EEF representation。

作者认为不输入 proprioception 可以减少：

- 对轨迹特定状态 trace 的过拟合；
- 不同机器人状态定义不一致；
- benchmark 中对绝对状态捷径的依赖。

但这也可能损失：

- 接触状态；
- 关节极限；
- 力矩；
- 不可见姿态；
- 遮挡下的机器人状态。

## 11.5 Appendix C.3：混频物理时间对齐

核心公式：

```text
H_b = round(tau * h_b)

t_b_i = i / h_b
```

并使用 sinusoidal physical-time encoding。

variable-length chunk 在 batch 中通过：

- padding；
- mask；
- 保留有效 token 的原生时间戳；

进行训练。

## 11.6 Appendix C.4：混频受控实验

使用同一组 20 Hz LIBERO 演示下采样构造 10/5 Hz 数据，因此没有新增专家轨迹。

`only 20Hz` 是上界参考，而不是数据更少的弱 baseline。

实验支持：

```text
物理时间编码确实缓解了 token index 与真实时间不一致
```

## 11.7 Appendix C.5：训练细节

### LaWM

```text
16×H100
100k steps
global batch 1024
AdamW
learning rate 3e-4
weight decay 1e-2
beta 1e-5
robot horizon 1.2 s
human horizon 0.4 s
```

### Policy integration

```text
64×H100
200k steps
global batch 1024
Action Expert lr 1e-4
other modules lr 3e-5
lambda_distill = 0.1
lambda_wm = 0.1
```

### 推理与延迟

```text
默认 10 个 denoising steps
A100
1,000 次重复
报告平均 wall-clock latency
```

## 11.8 Appendix D.1：LIBERO 协议

- 四个 suite；
- 去除失败 demonstration；
- 25k 后训练步数；
- batch 256；
- 40 tasks；
- 2,000 trials。

## 11.9 Appendix D.2：RoboTwin 协议

- 50 tasks；
- 2.5k clean demos；
- 25k randomized demos；
- 100k steps；
- batch 1024；
- 64×H100；
- 约 20 小时；
- 每任务每种 setting 100 trials。

## 11.10 Appendix D.3：真实机器人协议

三个任务的测试初始配置会先统一生成，再固定用于所有方法，以减少 baseline 之间的物理条件差异。

失败分析：

- Fast-WAM 在 Franka 任务中常出现距离估计不准；
- LingBot-VA 毛巾折叠受高延迟影响明显；
- LaWAM 对细微布料褶皱的 feature resolution 仍不足。

## 11.11 Appendix D.4：额外 LaWM rollout

Figure 10：

- 500 条 LIBERO 轨迹；
- 对比 rollout、真实未来和初始状态的 feature cosine similarity；
- 支持 LaWM 沿真实 latent dynamics 演化，而非停留在初始特征。

Figures 14–15：

- 单臂和双臂跨具身 rollout；
- 部分初始画面来自未见截图；
- 同一 latent action 在不同上下文中产生不同但连贯的 feature change。

这些结果是 representation-level evidence，不等于真实机器人 zero-shot 跨具身执行。

## 11.12 Appendix Figures 8–15 的作用

| Figure | 内容 | 能说明什么 |
|---|---|---|
| Fig. 8 | 完整 LIBERO chunk-level subgoal 序列 | 动作执行持续接近预测 subgoal |
| Fig. 9 | Franka 与 Quanta X1 平台 | 真实实验硬件设置 |
| Fig. 10 | 500 条 rollout similarity 曲线 | LaWM 不只是复制初始特征 |
| Fig. 11 | Pick-and-Place subgoal 可视化 | 机械臂与目标区域空间对应 |
| Fig. 12 | Drawer Opening subgoal 可视化 | 抓把手和抽拉过程的 chunk 指引 |
| Fig. 13 | Towel Folding 完整序列 | 长时程双臂布料任务 |
| Fig. 14 | 单臂跨具身 rollout | 共享 latent action 的上下文绑定 |
| Fig. 15 | 双臂跨具身 rollout | 未见场景中的 latent feature 演化 |

---

# 12. 论文局限与批判性分析

## 12.1 固定相机依赖

作者明确承认，在以下情况中 latent action 难以保持一致：

- 大幅相机移动；
- 第一人称剧烈抖动；
- humanoid 自运动；
- mobile robot 视角变化。

因为 IDM 看到的画面变化同时包含：

```text
物体真实运动
+
机器人自运动
+
相机运动
```

模型可能把 camera motion 错误编码为任务动作。

未来可能需要：

- ego-motion conditioning；
- 相机位姿；
- 3D latent；
- view-invariant representation；
- object-centric dynamics。

## 12.2 DINO feature 不等于完整物理状态

DINO 擅长：

- 语义；
- 空间对应；
- 物体级表示。

但不保证保留：

- 接触力；
- 摩擦；
- 质量；
- 深度；
- 遮挡后的状态；
- 细微布料褶皱；
- 关节受力；
- 滑动临界状态。

因此 latent feature prediction 高相似不一定意味着物理预测正确。

## 12.3 单一 subgoal 难以表示多模态未来

同一当前状态和指令可能有多条成功路径。LaWAM 默认预测一个 `z_hat` 和一个 `uT_hat`，并使用 L2 监督。

潜在问题：

- latent 平均；
- future feature averaging；
- 缺少风险估计；
- 无法比较多个计划；
- 对分叉任务可能保守或不稳定。

## 12.4 世界模型贡献没有完全因果隔离

`w/o WM` 下降支持 LaWM 有价值，但还需以下 matched controls：

- 相同参数量的 auxiliary future head；
- 随机 subgoal；
- shuffled subgoal；
- 停止梯度但保留参数；
- ground-truth future upper bound；
- 只增加相同数量 Transformer 层；
- latent token 直接输入 Action Expert。

否则增益可能混合：

- 更多参数；
- 更多监督；
- 更深网络；
- 更强 feature alignment；
- 真正的 predictive dynamics。

## 12.5 跨具身证据主要是定性

跨具身实验展示的是：

```text
同一 z + 不同初始截图
→ 不同 latent rollout
```

它支持上下文绑定，但没有直接证明：

```text
同一技能在未见机器人上零样本闭环成功
```

## 12.6 精度统计不足

论文大多报告单点成功率，没有：

- 多训练种子；
- 方差；
- 置信区间；
- 显著性；
- 配对试验差异。

LIBERO 98.6% 对 98.5% 的差距过小，不应过度解释。

## 12.7 训练并不轻量

“efficient”主要指推理，而不是训练：

- 4,500 小时视频；
- Stage 1 使用 16×H100；
- Stage 2 使用 64×H100；
- 200k policy-integration steps；
- RoboTwin 后训练还使用 64×H100 约 20 小时。

因此它是：

```text
高成本预训练
+
低延迟部署
```

## 12.8 人类视频贡献缺少独立消融

论文强调 1,500 小时第一人称人类视频提供 dynamics prior，但没有清晰报告：

```text
完整模型
vs
去掉全部人类视频
vs
等量增加机器人视频
```

因此无法量化人类视频的独立价值。

## 12.9 LaWM Stage 2 冻结策略不够明确

论文没有完整给出：

- optimizer parameter groups；
- `requires_grad`；
- stop-gradient 位置；
- `L_subgoal` 是否更新 LaWM；
- KI 是否同时隔离 VLM semantic stream；
- 各模块具体学习率归属。

这是复现时必须向作者确认的问题。

---

# 13. 最应该补充的实验

## 13.1 Latent-action 容量实验

- 扫描 `z` 的 token 数和维度；
- 报告 KL utilization；
- 测试 posterior collapse；
- 测试能否从 `z` 单独重建 `uT`；
- 测试打乱 `u` 后 LaWM 是否仍能预测未来。

目的是判断 `z` 是抽象动作还是高带宽 future code。

## 13.2 Posterior-prior gap

- 测量 `z_hat` 与 `z_teacher` 分布距离；
- 按任务与 horizon 报告 distillation error；
- 比较 deterministic prior、Gaussian prior、mixture prior；
- 使用 best-of-N latent samples；
- 测量 subgoal 多样性与任务成功率。

## 13.3 真正的 action-conditioned dynamics

给定同一当前状态和多个真实 action chunks：

```text
(u, a_candidate_1) → future_1
(u, a_candidate_2) → future_2
...
```

测试 LaWM 是否能准确区分后果，并用于：

- candidate ranking；
- MPC；
- safety filtering；
- value-guided selection。

## 13.4 统计可靠性

- 3–5 个训练种子；
- bootstrap confidence interval；
- 配对初始状态；
- 报告每任务 trial count；
- 统一 baseline 评测脚本；
- 统一 GPU、精度和采样器。

## 13.5 Moving-camera robustness

- 人为相机抖动；
- mobile robot；
- humanoid head camera；
- 外部相机位姿变化；
- 显式相机 pose conditioning；
- 3D 或 object-centric latent 对比。

## 13.6 数据贡献

- 无人类视频；
- 无机器人无标签视频；
- 不同数据量；
- 不同数据采样比例；
- 不同 horizon；
- 相同总时长下的人类/机器人替换。

---

# 14. 复现检查清单

论文当前没有完全给出的关键细节：

1. latent action 的 token 数、维度与 posterior 参数化；
2. Stage 2 中 LaWM 是否完全冻结；
3. Knowledge Insulation 的精确 stop-gradient 路径；
4. IDM teacher 使用 posterior sample 还是 posterior mean；
5. `L_aux` 在无人类状态标签视频上的 mask 方式；
6. `L_act` 的 flow interpolation 和 velocity target 方向；
7. action normalization、EEF 各维定义和 gripper 表示；
8. 每个数据集的采样比例；
9. Stage 1 robot/human 不同 horizon 的联合采样方式；
10. benchmark post-training 中各模块的冻结和学习率；
11. image augmentation 的具体范围；
12. optimizer 参数组、warmup、gradient clipping；
13. mixed-frequency padding 与 mask 实现；
14. latency测试的 batch、精度格式和编译设置；
15. 各 benchmark 的随机种子和 checkpoint 选择规则。

---

# 15. 最终评价

LaWAM 最重要的洞见是：

> 世界模型不一定要生成可以观看的未来视频；对于机器人策略，一个空间化、任务相关的未来视觉 latent 可能已经足够。

其核心贡献包括：

1. 将 latent-action model 的 forward decoder 重新利用为 LaWM；
2. 通过 IDM posterior 定义 latent-action 空间；
3. 通过 latent-action distillation 让 VLM 在看不到未来时驱动 LaWM；
4. 将当前与未来 DINO feature 作为动力学流送入 Alternate-DiT；
5. 用单次 latent subgoal prediction 替代迭代视频生成；
6. 使用物理时间编码处理混合控制频率。

证据最强的结论是：

```text
LaWAM 可以在保持较高控制成功率的同时，
显著降低 pixel-space WAM 的推理延迟。
```

证据尚不充分的结论是：

```text
LaWM 已经成为能够进行多方案反事实规划、
可靠跨具身迁移和精细物理预测的通用世界模型。
```

更准确的定位是：

```text
LaWAM =
  一个高效的、未来 latent 特征条件化 VLA

而不是 =
  已经完成通用 model-based planning 的机器人世界模型
```

如果目标是构建可部署的 WAM，最值得复用的是：

```text
latent action
  → 场景化 latent visual subgoal
  → action chunk
```

这个接口设计，而不是像素级未来生成。

---

<!-- 以下为已废弃的改写版问答记录，不参与 Markdown 渲染。

# 16. 已废弃的改写版问答记录

> 本节按时间顺序保存本次 LaWAM 论文讨论中的用户问题和对应回答。  
> 为保证 Markdown 可读性，原回答中的 LaTeX 公式被等价改写为纯文本公式，技术内容不删减。  
> 工具调用、文件检索过程和内部工作记录不属于面向用户的回答，因此不收录。

## Q1：解读 LaWAM 论文

**时间：2026-07-14 10:27（UTC+8）**

### 用户问题

> `c:\Users\r11379\Desktop\wm\wm_reward\LaWAM.pdf` 解读论文

### 回答

已完成结构化解读，并整理为可切换 Canvas，方便查看方法、实验和批判性分析：

[打开 LaWAM 论文解读](C:/Users/r11379/.cursor/projects/c-Users-r11379-Desktop-wm-wm-reward/canvases/LaWAM-paper-guide.canvas.tsx)

核心结论：

- 用 DINO 未来特征替代未来视频，显著降低 WAM 推理成本。
- 187 ms/chunk、LIBERO 98.6%、真实机器人 90.0%。
- 它更像“未来特征辅助策略”，不是经典的多候选世界模型规划。
- 精度领先幅度很小且缺少置信区间；移动相机、精细物理与跨具身验证仍是明显短板。

---

## Q2：结合 Figure 2 具体解释方法

**时间：2026-07-14 10:39（UTC+8）**

### 用户问题

> 结合图片具体解释一下方法

附图为论文 Figure 2：LaWAM 两阶段训练框架。

### 回答

这张图可以概括为：

> 先从视频中学习“视觉变化代码”，再让 VLM 预测这个代码；LaWM 将代码展开成当前场景下的未来视觉子目标，Action Expert 最后把子目标转换成机器人动作。

### Stage 1：学习 latent 世界模型

#### 1. 构造视觉转移

从视频取两个相隔固定物理时间的画面：

- `o1`：当前画面；
- `oT`：执行一段运动后的画面。

两帧经过冻结的 DINO：

```text
F1 = DINO(o1)
FT = DINO(oT)
```

图中的雪花表示 DINO 不参与训练。`F1`、`FT` 是保留空间布局的 patch 特征图，不是 RGB 图像。

#### 2. IDM 反推 latent action

Inverse Dynamics Model 同时观察 `F1` 和 `FT`，推断：

```text
z ~ q_phi(z | F1, FT)
```

图中绿色方块就是 latent action `z`。

它不是关节角、夹爪开合等真实动作，而是对“从当前画面变化到未来画面”所需转移的压缩编码。绿色方块只是示意，论文实际使用连续 latent action。

#### 3. LaWM 预测未来特征

LaWM 接收：

- 当前特征 `F1`；
- latent action `z`。

然后重建未来特征：

```text
FT_tilde = LaWM(F1, z)
```

训练目标主要是：

```text
||FT_tilde - FT||^2
```

因此，IDM 和 LaWM 形成了一个“视觉转移自编码器”：

```text
(F1, FT) → z
(F1, z)  → FT
```

训练完成后：

- IDM 用来给 Stage 2 产生教师 latent action；
- LaWM 被保留下来，成为部署时真正使用的世界模型；
- 部署时不需要未来画面，也不使用 IDM。

论文还加入 KL 正则和末端执行器状态预测，避免 `z` 只记忆外观差异。

### Stage 2：把 LaWM 接入 VLA

#### 1. VLM 预测 latent action

VLM 输入：

- 主视角 `o1`；
- “fold the towel”等语言指令；
- 其他相机视角；
- 一组 LA Query token。

LA Query 从 VLM 上下文中预测：

```text
z_hat = p_theta(z | o1, l)
```

训练时，Stage 1 的 IDM 可以根据真实 `F1`、`FT` 产生教师 `z`，于是使用蒸馏损失：

```text
L_distill = ||z_hat - z||^2
```

图中的虚线表示：让 VLM 在看不到未来的情况下，学会预测与 IDM 相同的视觉转移代码。

#### 2. LaWM 生成 latent visual subgoal

当前主视角再次经过 DINO 得到 `F1`。然后：

```text
FT_hat = LaWM(F1, z_hat)
```

`FT_hat` 就是图中的 Subgoal。

它表示“执行下一个动作块后，场景应该具有的 DINO 特征”，而不是生成出来的未来 RGB 图片。

训练时还会直接监督：

```text
L_wm = ||FT_hat - FT||^2
```

#### 3. Action Expert 生成真实动作

Action Expert 接收两类条件：

1. 语义流：VLM hidden states  
   告诉模型“任务是什么、操作哪个物体”。

2. 动力学流：当前特征 `F1` 与未来子目标 `FT_hat`  
   告诉模型“场景现在怎样、下一阶段应该变成怎样”。

Alternate-DiT 中交替使用：

- Self Attention：动作 token 之间协调；
- Inverse-Dynamics Attention：根据当前—未来特征推断所需运动；
- Semantic Attention：重新读取语言和视觉语义。

动作生成从图下方的 Noise 开始，经过约 10 个 flow-matching 去噪步骤，输出上方的连续 EEF action chunk。

Knowledge Insulation 的作用是允许 Action Expert 使用 LaWM 特征，但阻止动作损失随意破坏 LaWM 已学到的动力学表示。

### 推理时还保留什么

部署时没有真实未来画面，因此灰掉的部分全部消失：

```text
(o1, l)
  → z_hat
  → FT_hat
  → a1:T
```

具体过程是：

1. 当前图像和指令进入 VLM；
2. VLM 预测 `z_hat`；
3. LaWM 单次前向产生未来子目标 `FT_hat`；
4. Action Expert 生成动作块；
5. 执行动作，取得新画面，再重复上述过程。

以“fold the towel”为例，可以近似理解为：

- `z_hat`：抽象表示“完成下一步折叠变化”；
- LaWM：结合当前毛巾和双臂位置，预测折叠后应出现的空间特征；
- Action Expert：计算双臂如何移动、抓取和牵引才能接近这个特征子目标。

这里的 `z_hat` 并没有可直接解释的“折叠”标签，这只是理解其功能的类比。LaWAM 的核心价值正是：把抽象、跨具身的 latent action，展开成当前机器人和场景中的具体视觉子目标。

---

## Q3：Cursor 公式渲染有问题，重新输出

**时间：2026-07-14 10:45（UTC+8）**

### 用户问题

> Cursor 的公式渲染有问题，你重新输出一下。

### 回答

下面全部改用纯文本公式，避免 Cursor 的 LaTeX 渲染问题。

### 整体思路

```text
当前图像 + 指令
       ↓
预测 latent action
       ↓
LaWM 预测未来视觉特征
       ↓
Action Expert 生成真实动作
```

### Stage 1：学习 LaWM

#### 1. 提取当前与未来特征

- `o1`：当前画面；
- `oT`：一段时间后的真实画面；
- `F1`、`FT`：对应的 DINO 特征。

```text
F1 = DINO(o1)
FT = DINO(oT)
```

图中的雪花表示 DINO 是冻结的。

#### 2. IDM 反推 latent action

Inverse Dynamics Model 同时观察当前和未来：

```text
z = IDM(F1, FT)
```

其中 `z` 就是图中的绿色 Latent Action。

它不是关节角或末端执行器命令，而是一个连续编码，表示：

```text
“F1 需要发生什么视觉变化，才能变成 FT”
```

#### 3. LaWM 重建未来特征

```text
FT_pred = LaWM(F1, z)
```

训练时要求预测结果接近真实未来特征：

```text
L_world = MSE(FT_pred, FT)
```

完整训练目标还包括：

```text
L_stage1 =
    L_world
  + L_aux_state
  + beta * L_KL
```

作用分别是：

- `L_world`：学习未来视觉变化；
- `L_aux_state`：让 `z` 包含机器人运动信息；
- `L_KL`：让 latent action 空间更平滑、容易预测。

因此 Stage 1 本质上是：

```text
(F1, FT) --IDM--> z
(F1, z)  --LaWM--> FT_pred
```

训练完成后：

- IDM 只负责为 Stage 2 提供教师信号；
- LaWM 被保留下来作为世界模型；
- 推理时不再需要 IDM 和真实未来画面。

### Stage 2：把 LaWM 接入 VLA

#### 1. VLM 预测 latent action

VLM 输入：

```text
当前主视角
+ 语言指令
+ 其他相机视角
+ LA Query
```

然后预测：

```text
z_hat = PolicyPrior(current_images, instruction)
```

训练时，Stage 1 的 IDM 会产生教师 latent action：

```text
z_teacher = IDM(F1, FT)
```

两者通过蒸馏损失对齐：

```text
L_distill = MSE(z_hat, z_teacher)
```

因此，虚线箭头的含义是：

```text
教 VLM 在看不到未来 FT 的情况下，
预测出与 IDM 相似的 latent action。
```

#### 2. LaWM 预测视觉子目标

```text
FT_hat = LaWM(F1, z_hat)
```

`FT_hat` 就是图中的 Subgoal。

它不是未来 RGB 图像，而是：

```text
“执行下一个动作块后，场景应该达到的 DINO 特征状态”
```

训练时还有子目标损失：

```text
L_subgoal = MSE(FT_hat, FT)
```

#### 3. Action Expert 生成真实动作

Action Expert 获得两类信息。

语义信息：

```text
VLM hidden states
```

它描述任务意图，例如“折叠毛巾”。

动力学信息：

```text
current feature F1
+ predicted future feature FT_hat
```

它描述场景需要发生什么变化。

因此可以把 Action Expert 理解为：

```text
actions = ActionExpert(
    semantic_context,
    F1,
    FT_hat,
    noise
)
```

Action Expert 从噪声动作开始，通过 flow-matching 逐步去噪，最终输出一段连续动作。

Alternate-DiT 内部交替执行：

```text
Self Attention
→ Inverse-Dynamics Attention
→ Self Attention
→ Semantic Attention
```

分别负责：

- 协调动作序列内部关系；
- 根据当前状态和未来子目标推断运动；
- 重新读取任务语义。

#### 4. Knowledge Insulation

图中 `Knowledge Insulation & Concat` 的作用是：

```text
将 F1 和 FT_hat 组合成动力学条件
+
阻止 Action Expert 的梯度破坏 LaWM
```

也就是说，Action Expert 可以使用 LaWM 的预测，但不能通过动作损失随意改写 LaWM 已学到的动力学知识。

Stage 2 总损失为：

```text
L_stage2 =
    0.1 * L_distill
  + 0.1 * L_subgoal
  + L_action
```

### 部署时的完整流程

推理时不存在真实未来画面 `oT`，也不使用 IDM：

```text
1. F1 = DINO(o1)

2. z_hat = PolicyPrior(o1, instruction)

3. FT_hat = LaWM(F1, z_hat)

4. action_chunk = ActionExpert(
       VLM_context,
       F1,
       FT_hat,
       noise
   )

5. 执行动作块，获取新画面，然后重复
```

简化表示为：

```text
当前观测 + 指令
        ↓
latent action z_hat
        ↓
未来特征子目标 FT_hat
        ↓
真实机器人动作块
```

以“fold the towel”为例：

```text
z_hat：
抽象表示下一阶段需要发生的折叠变化

LaWM：
结合当前毛巾和机械臂位置，
生成折叠后应该达到的视觉特征

Action Expert：
计算双臂应该如何抓取、移动和牵引
```

核心点是：

```text
latent action 描述“发生什么变化”

LaWM 描述“这个变化在当前场景中是什么样子”

Action Expert 描述“机器人具体应该怎么做”
```

---

## Q4：IDM 的作用是什么

**时间：2026-07-14 11:02（UTC+8）**

### 用户问题

> IDM 的作用是什么？

### 回答

IDM（Inverse Dynamics Model，逆动力学模型）的核心作用是：

> 根据“变化前”和“变化后”的视觉特征，反推出导致这次变化的 latent action。

```text
输入：当前特征 F1 + 未来特征 FT
输出：latent action z

z = IDM(F1, FT)
```

它在 LaWAM 中有三个具体用途：

1. 从无动作标注视频中提取运动信息  
   只要有前后两帧，就能推断一个描述视觉转移的 `z`，不需要真实机器人动作标签。

2. 帮助训练 LaWM

```text
z = IDM(F1, FT)
FT_pred = LaWM(F1, z)
```

LaWM 必须根据 `F1 + z` 恢复 `FT`，因此 `z` 被迫编码“发生了什么变化”。

3. 作为 Stage 2 的教师

```text
z_teacher = IDM(F1, FT)
z_hat     = VLM(current_image, instruction)
```

通过蒸馏，让 VLM 在看不到未来画面时也能预测正确的 latent action。

需要特别注意：

- IDM 输出的不是关节角或机器人控制命令；
- 它输出的是抽象视觉转移代码；
- IDM 只在训练时使用；
- 部署时没有真实未来 `FT`，因此 IDM 会被移除。

一句话概括：

```text
IDM：从已经发生的视觉变化中提取“变化代码”
LaWM：根据变化代码预测未来
VLM：学习在未来尚未发生时预测这个变化代码
```

---

## Q5：IDM 是否需要训练，latent action 由谁 embedding

**时间：2026-07-14 11:15（UTC+8）**

### 用户问题

> 这个 IDE 这么叼？不需要训练吗，latent action 是经过谁 embedding 的呢？

### 回答

这里应该是 **IDM**。它当然需要训练，并不是现成就会。

### 谁在 embedding latent action

分两层：

1. DINO 负责把图像 embedding 成视觉特征；
2. IDM 负责把“前后视觉特征的变化”embedding 成 latent action。

```text
o1, oT
  ↓ 冻结的 DINO
F1, FT
  ↓ 可训练的 IDM
latent action z
```

IDM 本身就是 latent-action encoder，并没有另一个模型先把真实动作 embedding 进去。

### IDM 如何训练

Stage 1 中，IDM 和 LaWM 联合训练：

```text
z       = IDM(F1, FT)
FT_pred = LaWM(F1, z)
```

然后要求：

```text
FT_pred 接近真实 FT
```

完整目标近似为：

```text
L =
    MSE(FT_pred, FT)
  + MSE(predicted_robot_state, future_robot_state)
  + beta * KL(q(z | F1, FT), Normal(0, I))
```

反向传播时：

- DINO：冻结；
- IDM：更新；
- LaWM：更新；
- 辅助状态预测器：更新。

论文使用连续 latent action。IDM 实际学习的是一个后验分布：

```text
q_phi(z | F1, FT)
```

可以理解为 IDM 输出分布参数，再从中得到 `z`。

### 为什么它看起来这么强

因为 IDM 训练时同时看到了“变化前”和“变化后”：

```text
IDM 看 F1 和 FT：
已经知道结果，再总结发生了什么变化
```

所以它做的不是困难的未来预测，而是将已经发生的变化压缩成代码。类似于同时看两张图片，再提取它们之间的运动差异。

真正困难的是部署阶段：未来 `FT` 还不存在，此时由 VLM 预测：

```text
z_hat = VLM(current_images, instruction, LA_Query)
```

Stage 2 用 IDM 产生的 `z` 教 VLM：

```text
z_teacher = IDM(F1, FT)
z_hat     = VLM(o1, instruction)

L_distill = MSE(z_hat, z_teacher)
```

所以：

```text
IDM：看过未来后提取标准答案
VLM：只看当前和指令，学习猜这个答案
LaWM：把答案展开成未来视觉子目标
```

### latent action 会不会“作弊”

这是一个真实风险。因为 IDM 看到了 `FT`，它可能把大量未来信息直接塞入 `z`，而不是真正提取可泛化的动力学。

论文使用以下手段限制它：

- latent bottleneck；
- KL 正则，使 `z` 接近标准正态分布；
- 辅助机器人状态预测，鼓励 `z` 表示运动；
- 不同裁剪和颜色增强，减少对固定像素布局的记忆；
- LaWM 必须结合 `F1` 才能恢复未来。

但论文没有清楚报告 `z` 的 token 数和具体维度，因此“`z` 到底压缩得够不够、是否携带过多未来信息”仍是方法的一个验证缺口。

---

## Q6：IDM 与 VLM 输出的 latent action 能否一致

**时间：2026-07-14 15:44（UTC+8）**

### 用户问题

> LaWM 在训练的时候，输入是 IDM 出的 latent action 和 F1；在推理的时候，怎么变成了 VLM 出 latent action？这两个能“一样”吗？

### 回答

如果只训练 Stage 1，推理时直接把 IDM 换成 VLM，通常会产生严重分布偏移。LaWAM 的 Stage 2 就是专门解决这个替换问题的。

### Stage 1：IDM 定义 latent 空间

```text
z_teacher = IDM(F1, FT)
FT_pred   = LaWM(F1, z_teacher)
```

IDM 看到了真实未来 `FT`，因此产生的 `z_teacher` 相当于“标准答案”。LaWM 由此学会如何解释这种 latent action。

### Stage 2：训练 VLM 模仿 IDM

VLM 并不是推理时突然接管。在 Stage 2 训练期间，LaWM 接收的就已经是 VLM 输出：

```text
z_hat  = VLM(current_images, instruction, LA_Query)
FT_hat = LaWM(F1, z_hat)
```

同时使用两层约束。

第一层：数值对齐

```text
L_distill = MSE(z_hat, z_teacher)
```

VLM 输出经过 query aggregation/projection，维度与 `z_teacher` 完全相同，并直接逐维回归 IDM 的 latent action。

第二层：功能对齐

```text
L_subgoal = MSE(LaWM(F1, z_hat), FT)
```

即使 `z_hat` 没有与 `z_teacher` 数值完全一致，只要经过 LaWM 后能预测正确未来特征，也仍然是有效 latent action。

因此训练目标是：

```text
VLM 输出不仅要“长得像”IDM 输出，
还要能够正确驱动同一个 LaWM。
```

### 两者能完全一样吗

分三个层面：

```text
张量形状：一样，由网络结构强制保证
latent 坐标系：一样，由蒸馏损失直接对齐
具体数值：通常不可能完全一样
```

原因是两者掌握的信息不同：

```text
IDM 输入：F1 + 真实 FT
VLM 输入：当前图像 + 指令
```

IDM 已经看到了结果，VLM 只能预测结果。因此 `z_hat` 只是 `z_teacher` 的近似预测。

LaWM 实际需要的也不是逐位完全相同，而是：

```text
LaWM(F1, z_hat)
  ≈ LaWM(F1, z_teacher)
  ≈ FT
```

### 仍然存在的核心问题

如果同一个当前状态和指令可能对应多种未来：

```text
当前状态 + “把杯子放进篮子”
→ 可以从左边抓
→ 可以从右边抓
→ 可以采用不同运动轨迹
```

IDM 根据真实发生的那条轨迹输出具体 `z_teacher`，但 VLM 无法提前知道数据采集时选择了哪条轨迹。使用 MSE 蒸馏时，VLM 可能输出多个 latent action 的平均值。

这就是典型的 posterior-prior gap：

```text
IDM：拥有未来信息的 posterior
VLM：没有未来信息的 learned prior
```

论文通过以下方式缓解：

- Stage 1 的 KL 正则让 latent 空间更平滑；
- 直接蒸馏 `z_hat`；
- 再监督 LaWM 输出的 `FT_hat`；
- Action Expert 在训练时就使用预测的 `FT_hat`；
- 每执行一个 action chunk 就重新观察和预测。

但它没有彻底解决多模态未来问题。论文没有采用多 latent 采样、best-of-N 或候选未来规划。

最准确的理解是：

```text
IDM 定义 latent action 的“语言”
VLM 学习说这种语言
LaWM 检查 VLM 说出的内容能否产生正确未来
```

所以两者不是天然一样，而是通过 Stage 2 的“latent 数值蒸馏 + 未来特征监督”被强行对齐。

---

## Q7：Stage 2 中 LaWM 是否冻结

**时间：2026-07-14 17:45（UTC+8）**

### 用户问题

> Stage 2 在训练的时候，LaWM 是冻住的吧？

### 回答

不能简单说“完全冻住”。论文明确的是：

> Stage 2 中，Action Expert 的动作损失不能反向更新 LaWM；但论文没有明确声明 LaWM 对所有损失完全冻结。

梯度关系更可能是：

```text
L_distill
  → 更新 VLM / policy prior

L_wm = MSE(LaWM(F1, z_hat), FT)
  → 更新 policy prior
  → 可能继续微调 LaWM

L_action
  → 更新 Action Expert
  × 不允许经过 KI 破坏 LaWM
```

也就是：

```text
LaWM 对 L_action 是“冻结/stop-gradient”的
LaWM 对 L_wm 不一定冻结
```

支持这一判断的依据：

1. 论文只明确称 DINO 是 frozen encoder，没有明确称 Stage 2 的 LaWM frozen。
2. Stage 2 包含子目标损失：

```text
L_wm = MSE(FT_hat, FT)
FT_hat = LaWM(F1, z_hat)
```

3. 作者特意使用 Knowledge Insulation，表述为：

```text
防止 Action Expert 的梯度覆盖 LaWM 的预训练动力学
```

如果 LaWM 对所有损失都完全冻结，这个说法会有些多余。

4. 附录称 Action Expert 使用 `1e-4` 学习率，“其他模块”使用 `3e-5`，也倾向于说明部分上游模块会继续训练。

不过，全冻结在数学上也是可行的：

```text
固定 LaWM 参数
L_wm 经过 LaWM 的计算图
只更新产生 z_hat 的 VLM
```

此时 LaWM 相当于固定的可微判别器，强迫 VLM 输出能被它正确解码的 latent action。

因此，基于当前 arXiv v1，最严谨的结论是：

```text
确定：L_action 不更新 LaWM
很可能：L_wm 可以小学习率微调 LaWM
不确定：作者是否在实现中彻底冻结了 LaWM 参数
```

论文没有给出 optimizer 参数组或明确的 `requires_grad` 配置，所以无法仅凭正文断言“Stage 2 的 LaWM 完全冻住”。

---

## Q8：将论文解读、Loss、实验和附录保存为 Markdown

**时间：2026-07-14 17:52（UTC+8）**

### 用户问题

> OK，把论文解读和我们讨论的内容写成 MD 文档存在本地，把每个 Stage 训练的 Loss 写清楚每个部分具体是什么。还有实验部分和附录。

### 回答

已完成并打开：

[LaWAM-论文解读与讨论.md](C:/Users/r11379/Desktop/wm/wm_reward/LaWAM-论文解读与讨论.md)

内容包括：

- Stage 1/2 完整流程与各项 Loss；
- IDM、latent action、LaWM、VLM 的关系；
- Stage 2 冻结与梯度流争议；
- Flow Matching 动作损失；
- 混合控制频率与时间编码；
- LIBERO、RoboTwin、真实机器人实验；
- 全部附录解读；
- 局限、复现缺口及建议实验。

所有公式均使用纯文本，避免 Cursor 渲染问题。

---

## Q9：补充论文时间和作者单位

**时间：2026-07-15 09:36（UTC+8）**

### 用户问题

> MD 文档里面写一下论文的时间和发表单位。

### 回答

已补充：

- 首次公开时间：2026-06-14；
- 发表平台及预印本状态；
- 清华大学、吉林大学、南开大学、北京大学、哈尔滨工业大学、中关村学院、Striding.AI、无问芯穹（Infinigence AI）等全部作者单位。

---

## Q10：完整保存所有问题与回答

**时间：2026-07-15 11:20（UTC+8）**

### 用户问题

> 我问的问题和你的回答要完整保存在 MD 中。

### 回答

已将本次 LaWAM 论文讨论中所有用户问题和对应的面向用户回答，按照时间顺序完整追加到本文档第 16 节。

其中：

- 保留了每个问题的原始含义与完整技术回答；
- 原 LaTeX 公式已等价改写为纯文本，避免 Cursor 渲染异常；
- 不收录工具调用、文件检索和内部工作记录；
- 后续与 LaWAM 相关的新问题及回答也应继续追加到本节。

以上为已废弃的改写版问答记录。 -->

---

# 16. 完整问答逐字记录

> 以下只保存用户原话和当时面向用户的完整回答原文，不总结、不改写、不润色。  
> 为防止原回答中的 Markdown 与 LaTeX 被重新渲染，每条消息使用四反引号代码块原样保存。

## 消息 1：用户

````text
c:\Users\r11379\Desktop\wm\wm_reward\LaWAM.pdf 解读论文
````

## 消息 2：回答

````markdown
已完成结构化解读，并整理为可切换 Canvas，方便查看方法、实验和批判性分析：

[打开 LaWAM 论文解读](C:/Users/r11379/.cursor/projects/c-Users-r11379-Desktop-wm-wm-reward/canvases/LaWAM-paper-guide.canvas.tsx)

核心结论：
- 用 DINO 未来特征替代未来视频，显著降低 WAM 推理成本。
- 187 ms/chunk、LIBERO 98.6%、真实机器人 90.0%。
- 它更像“未来特征辅助策略”，不是经典的多候选世界模型规划。
- 精度领先幅度很小且缺少置信区间；移动相机、精细物理与跨具身验证仍是明显短板。
````

## 消息 3：用户

````text
结合图片具体解释一下方法
````

该问题同时附带论文 Figure 2 图片。

## 消息 4：回答

````markdown
这张图可以概括为：

> 先从视频中学习“视觉变化代码”，再让 VLM 预测这个代码；LaWM 将代码展开成当前场景下的未来视觉子目标，Action Expert 最后把子目标转换成机器人动作。

## Stage 1：学习 latent 世界模型

### 1. 构造视觉转移

从视频取两个相隔固定物理时间的画面：

- \(o_1\)：当前画面
- \(o_T\)：执行一段运动后的画面

两帧经过冻结的 DINO：

\[
F_1=\mathrm{DINO}(o_1),\qquad F_T=\mathrm{DINO}(o_T)
\]

图中的雪花表示 DINO 不参与训练。\(F_1,F_T\) 是保留空间布局的 patch 特征图，不是 RGB 图像。

### 2. IDM 反推 latent action

Inverse Dynamics Model 同时观察 \(F_1\) 和 \(F_T\)，推断：

\[
z\sim q_\phi(z\mid F_1,F_T)
\]

图中绿色方块就是 latent action \(z\)。

它不是关节角、夹爪开合等真实动作，而是对“从当前画面变化到未来画面”所需转移的压缩编码。绿色方块只是示意，论文实际使用连续 latent action。

### 3. LaWM 预测未来特征

LaWM 接收：

- 当前特征 \(F_1\)
- latent action \(z\)

然后重建未来特征：

\[
\tilde F_T=\mathrm{LaWM}(F_1,z)
\]

训练目标主要是：

\[
\|\tilde F_T-F_T\|_2^2
\]

因此，IDM 和 LaWM 形成了一个“视觉转移自编码器”：

\[
(F_1,F_T)\rightarrow z,\qquad(F_1,z)\rightarrow F_T
\]

训练完成后：

- IDM 用来给 Stage 2 产生教师 latent action；
- LaWM 被保留下来，成为部署时真正使用的世界模型；
- 部署时不需要未来画面，也不使用 IDM。

论文还加入 KL 正则和末端执行器状态预测，避免 \(z\) 只记忆外观差异。

---

## Stage 2：把 LaWM 接入 VLA

### 1. VLM 预测 latent action

VLM 输入：

- 主视角 \(o_1\)
- “fold the towel”等语言指令
- 其他相机视角
- 一组 LA Query token

LA Query 从 VLM 上下文中预测：

\[
\hat z=p_\theta(z\mid o_1,l)
\]

训练时，Stage 1 的 IDM 可以根据真实 \(F_1,F_T\) 产生教师 \(z\)，于是使用蒸馏损失：

\[
L_{\text{distill}}=\|\hat z-z\|_2^2
\]

这条虚线表示：让 VLM 在看不到未来的情况下，学会预测与 IDM 相同的视觉转移代码。

### 2. LaWM 生成 latent visual subgoal

当前主视角再次经过 DINO 得到 \(F_1\)。然后：

\[
\hat F_T=\mathrm{LaWM}(F_1,\hat z)
\]

\(\hat F_T\) 就是图中的 Subgoal。

它表示“执行下一个动作块后，场景应该具有的 DINO 特征”，而不是生成出来的未来 RGB 图片。

训练时还会直接监督：

\[
L_{\text{wm}}=\|\hat F_T-F_T\|_2^2
\]

### 3. Action Expert 生成真实动作

Action Expert 接收两类条件：

1. 语义流：VLM hidden states  
   告诉模型“任务是什么、操作哪个物体”。

2. 动力学流：当前特征 \(F_1\) 与未来子目标 \(\hat F_T\)  
   告诉模型“场景现在怎样、下一阶段应该变成怎样”。

Alternate-DiT 中交替使用：

- Self Attention：动作 token 之间协调；
- Inverse-Dynamics Attention：根据当前—未来特征推断所需运动；
- Semantic Attention：重新读取语言和视觉语义。

动作生成从图下方的 Noise 开始，经过约 10 个 flow-matching 去噪步骤，输出上方的连续 EEF action chunk。

Knowledge Insulation 的作用是允许 Action Expert 使用 LaWM 特征，但阻止动作损失随意破坏 LaWM 已学到的动力学表示。

---

## 推理时还保留什么？

部署时没有真实未来画面，因此灰掉的部分全部消失：

\[
(o_1,l)
\rightarrow \hat z
\rightarrow \hat F_T
\rightarrow a_{1:T}
\]

具体过程是：

1. 当前图像和指令进入 VLM；
2. VLM 预测 \(\hat z\)；
3. LaWM 单次前向产生未来子目标 \(\hat F_T\)；
4. Action Expert 生成动作块；
5. 执行动作，取得新画面，再重复上述过程。

---

以“fold the towel”为例，可以近似理解为：

- \(\hat z\)：抽象表示“完成下一步折叠变化”；
- LaWM：结合当前毛巾和双臂位置，预测折叠后应出现的空间特征；
- Action Expert：计算双臂如何移动、抓取和牵引才能接近这个特征子目标。

这里的 \(\hat z\) 并没有可直接解释的“折叠”标签，这只是理解其功能的类比。LaWAM 的核心价值正是：把抽象、跨具身的 latent action，展开成当前机器人和场景中的具体视觉子目标。
````

## 消息 5：用户

````text
cursor的公式渲染有问题，你重新输出一下
````

## 消息 6：回答

````markdown
下面全部改用纯文本公式，避免 Cursor 的 LaTeX 渲染问题。

## 整体思路

```text
当前图像 + 指令
       ↓
预测 latent action
       ↓
LaWM 预测未来视觉特征
       ↓
Action Expert 生成真实动作
```

## Stage 1：学习 LaWM

### 1. 提取当前与未来特征

- `o1`：当前画面
- `oT`：一段时间后的真实画面
- `F1`、`FT`：对应的 DINO 特征

```text
F1 = DINO(o1)
FT = DINO(oT)
```

图中的雪花表示 DINO 是冻结的。

### 2. IDM 反推 latent action

Inverse Dynamics Model 同时观察当前和未来：

```text
z = IDM(F1, FT)
```

其中 `z` 就是图中的绿色 Latent Action。

它不是关节角或末端执行器命令，而是一个连续编码，表示：

```text
“F1 需要发生什么视觉变化，才能变成 FT”
```

### 3. LaWM 重建未来特征

```text
FT_pred = LaWM(F1, z)
```

训练时要求预测结果接近真实未来特征：

```text
L_world = MSE(FT_pred, FT)
```

完整训练目标还包括：

```text
L_stage1 =
    L_world
  + L_aux_state
  + beta * L_KL
```

作用分别是：

- `L_world`：学习未来视觉变化；
- `L_aux_state`：让 `z` 包含机器人运动信息；
- `L_KL`：让 latent action 空间更平滑、容易预测。

因此 Stage 1 本质上是：

```text
(F1, FT) --IDM--> z
(F1, z)  --LaWM--> FT_pred
```

训练完成后：

- IDM 只负责为 Stage 2 提供教师信号；
- LaWM 被保留下来作为世界模型；
- 推理时不再需要 IDM 和真实未来画面。

---

## Stage 2：把 LaWM 接入 VLA

### 1. VLM 预测 latent action

VLM 输入：

```text
当前主视角
+ 语言指令
+ 其他相机视角
+ LA Query
```

然后预测：

```text
z_hat = PolicyPrior(current_images, instruction)
```

训练时，Stage 1 的 IDM 会产生教师 latent action：

```text
z_teacher = IDM(F1, FT)
```

两者通过蒸馏损失对齐：

```text
L_distill = MSE(z_hat, z_teacher)
```

因此，虚线箭头的含义是：

```text
教 VLM 在看不到未来 FT 的情况下，
预测出与 IDM 相似的 latent action。
```

### 2. LaWM 预测视觉子目标

```text
FT_hat = LaWM(F1, z_hat)
```

`FT_hat` 就是图中的 Subgoal。

它不是未来 RGB 图像，而是：

```text
“执行下一个动作块后，场景应该达到的 DINO 特征状态”
```

训练时还有子目标损失：

```text
L_subgoal = MSE(FT_hat, FT)
```

### 3. Action Expert 生成真实动作

Action Expert 获得两类信息。

语义信息：

```text
VLM hidden states
```

它描述任务意图，例如“折叠毛巾”。

动力学信息：

```text
current feature F1
+ predicted future feature FT_hat
```

它描述场景需要发生什么变化。

因此可以把 Action Expert 理解为：

```text
actions = ActionExpert(
    semantic_context,
    F1,
    FT_hat,
    noise
)
```

Action Expert 从噪声动作开始，通过 flow-matching 逐步去噪，最终输出一段连续动作。

Alternate-DiT 内部交替执行：

```text
Self Attention
→ Inverse-Dynamics Attention
→ Self Attention
→ Semantic Attention
```

分别负责：

- 协调动作序列内部关系；
- 根据当前状态和未来子目标推断运动；
- 重新读取任务语义。

### 4. Knowledge Insulation

图中 `Knowledge Insulation & Concat` 的作用是：

```text
将 F1 和 FT_hat 组合成动力学条件
+
阻止 Action Expert 的梯度破坏 LaWM
```

也就是说，Action Expert 可以使用 LaWM 的预测，但不能通过动作损失随意改写 LaWM 已学到的动力学知识。

Stage 2 总损失为：

```text
L_stage2 =
    0.1 * L_distill
  + 0.1 * L_subgoal
  + L_action
```

---

## 部署时的完整流程

推理时不存在真实未来画面 `oT`，也不使用 IDM：

```text
1. F1 = DINO(o1)

2. z_hat = PolicyPrior(o1, instruction)

3. FT_hat = LaWM(F1, z_hat)

4. action_chunk = ActionExpert(
       VLM_context,
       F1,
       FT_hat,
       noise
   )

5. 执行动作块，获取新画面，然后重复
```

简化表示为：

```text
当前观测 + 指令
        ↓
latent action z_hat
        ↓
未来特征子目标 FT_hat
        ↓
真实机器人动作块
```

以“fold the towel”为例：

```text
z_hat：
抽象表示下一阶段需要发生的折叠变化

LaWM：
结合当前毛巾和机械臂位置，
生成折叠后应该达到的视觉特征

Action Expert：
计算双臂应该如何抓取、移动和牵引
```

核心点是：

```text
latent action 描述“发生什么变化”

LaWM 描述“这个变化在当前场景中是什么样子”

Action Expert 描述“机器人具体应该怎么做”
```
````

## 消息 7：用户

````text
IDM的作用是什么
````

## 消息 8：回答

````markdown
IDM（Inverse Dynamics Model，逆动力学模型）的核心作用是：

> 根据“变化前”和“变化后”的视觉特征，反推出导致这次变化的 latent action。

```text
输入：当前特征 F1 + 未来特征 FT
输出：latent action z

z = IDM(F1, FT)
```

它在 LaWAM 中有三个具体用途：

1. 从无动作标注视频中提取运动信息  
   只要有前后两帧，就能推断一个描述视觉转移的 `z`，不需要真实机器人动作标签。

2. 帮助训练 LaWM

```text
z = IDM(F1, FT)
FT_pred = LaWM(F1, z)
```

LaWM 必须根据 `F1 + z` 恢复 `FT`，因此 `z` 被迫编码“发生了什么变化”。

3. 作为 Stage 2 的教师

```text
z_teacher = IDM(F1, FT)
z_hat     = VLM(current_image, instruction)
```

通过蒸馏，让 VLM 在看不到未来画面时也能预测正确的 latent action。

需要特别注意：

- IDM 输出的不是关节角或机器人控制命令；
- 它输出的是抽象视觉转移代码；
- IDM 只在训练时使用；
- 部署时没有真实未来 `FT`，因此 IDM 会被移除。

一句话概括：

```text
IDM：从已经发生的视觉变化中提取“变化代码”
LaWM：根据变化代码预测未来
VLM：学习在未来尚未发生时预测这个变化代码
```
````

## 消息 9：用户

````text
这个IDE这么叼？不需要训练吗，latent action是经过谁embedding的呢
````

## 消息 10：回答

````markdown
你这里应该是 **IDM**。它当然需要训练，并不是现成就会。

## 谁在 embedding latent action？

分两层：

1. DINO 负责把图像 embedding 成视觉特征；
2. IDM 负责把“前后视觉特征的变化”embedding 成 latent action。

```text
o1, oT
  ↓ 冻结的 DINO
F1, FT
  ↓ 可训练的 IDM
latent action z
```

IDM 本身就是 latent-action encoder，并没有另一个模型先把真实动作 embedding 进去。

## IDM 如何训练？

Stage 1 中，IDM 和 LaWM 联合训练：

```text
z       = IDM(F1, FT)
FT_pred = LaWM(F1, z)
```

然后要求：

```text
FT_pred 接近真实 FT
```

完整目标近似为：

```text
L =
    MSE(FT_pred, FT)
  + MSE(predicted_robot_state, future_robot_state)
  + beta * KL(q(z | F1, FT), Normal(0, I))
```

反向传播时：

- DINO：冻结；
- IDM：更新；
- LaWM：更新；
- 辅助状态预测器：更新。

论文使用连续 latent action。IDM 实际学习的是一个后验分布：

```text
q_phi(z | F1, FT)
```

可以理解为 IDM 输出分布参数，再从中得到 `z`。

## 为什么它看起来这么强？

因为 IDM 训练时同时看到了“变化前”和“变化后”：

```text
IDM 看 F1 和 FT：
已经知道结果，再总结发生了什么变化
```

所以它做的不是困难的未来预测，而是将已经发生的变化压缩成代码。类似于同时看两张图片，再提取它们之间的运动差异。

真正困难的是部署阶段：未来 `FT` 还不存在，此时由 VLM 预测：

```text
z_hat = VLM(current_images, instruction, LA_Query)
```

Stage 2 用 IDM 产生的 `z` 教 VLM：

```text
z_teacher = IDM(F1, FT)
z_hat     = VLM(o1, instruction)

L_distill = MSE(z_hat, z_teacher)
```

所以：

```text
IDM：看过未来后提取标准答案
VLM：只看当前和指令，学习猜这个答案
LaWM：把答案展开成未来视觉子目标
```

## latent action 会不会“作弊”？

这是一个真实风险。因为 IDM 看到了 `FT`，它可能把大量未来信息直接塞入 `z`，而不是真正提取可泛化的动力学。

论文使用以下手段限制它：

- latent bottleneck；
- KL 正则，使 `z` 接近标准正态分布；
- 辅助机器人状态预测，鼓励 `z` 表示运动；
- 不同裁剪和颜色增强，减少对固定像素布局的记忆；
- LaWM 必须结合 `F1` 才能恢复未来。

但论文没有清楚报告 `z` 的 token 数和具体维度，因此“`z` 到底压缩得够不够、是否携带过多未来信息”仍是方法的一个验证缺口。
````

## 消息 11：用户

````text
LaWM在训练的时候，输入是IDM出的latent action和F1，在推理的时候，怎么变成了VLM出latent action？这两个能“一样”吗
````

## 消息 12：回答

````markdown
你的疑问是对的：如果只训练 Stage 1，推理时直接把 IDM 换成 VLM，通常会产生严重分布偏移。LaWAM 的 Stage 2 就是专门解决这个替换问题的。

## Stage 1：IDM 定义 latent 空间

```text
z_teacher = IDM(F1, FT)
FT_pred   = LaWM(F1, z_teacher)
```

IDM 看到了真实未来 `FT`，因此产生的 `z_teacher` 相当于“标准答案”。LaWM 由此学会如何解释这种 latent action。

## Stage 2：训练 VLM 模仿 IDM

VLM 并不是推理时突然接管。在 Stage 2 训练期间，LaWM 接收的就已经是 VLM 输出：

```text
z_hat  = VLM(current_images, instruction, LA_Query)
FT_hat = LaWM(F1, z_hat)
```

同时使用两层约束。

第一层：数值对齐

```text
L_distill = MSE(z_hat, z_teacher)
```

VLM 输出经过 query aggregation/projection，维度与 `z_teacher` 完全相同，并直接逐维回归 IDM 的 latent action。

第二层：功能对齐

```text
L_subgoal = MSE(LaWM(F1, z_hat), FT)
```

即使 `z_hat` 没有与 `z_teacher` 数值完全一致，只要经过 LaWM 后能预测正确未来特征，也仍然是有效 latent action。

因此训练目标是：

```text
VLM 输出不仅要“长得像”IDM 输出，
还要能够正确驱动同一个 LaWM。
```

## 两者能完全一样吗？

分三个层面：

```text
张量形状：一样，由网络结构强制保证
latent 坐标系：一样，由蒸馏损失直接对齐
具体数值：通常不可能完全一样
```

原因是两者掌握的信息不同：

```text
IDM 输入：F1 + 真实 FT
VLM 输入：当前图像 + 指令
```

IDM 已经看到了结果，VLM 只能预测结果。因此 `z_hat` 只是 `z_teacher` 的近似预测。

LaWM 实际需要的也不是逐位完全相同，而是：

```text
LaWM(F1, z_hat) ≈ LaWM(F1, z_teacher) ≈ FT
```

## 仍然存在的核心问题

如果同一个当前状态和指令可能对应多种未来：

```text
当前状态 + “把杯子放进篮子”
→ 可以从左边抓
→ 可以从右边抓
→ 可以采用不同运动轨迹
```

IDM 根据真实发生的那条轨迹输出具体 `z_teacher`，但 VLM 无法提前知道数据采集时选择了哪条轨迹。使用 MSE 蒸馏时，VLM 可能输出多个 latent action 的平均值。

这就是典型的 posterior-prior gap：

```text
IDM：拥有未来信息的 posterior
VLM：没有未来信息的 learned prior
```

论文通过以下方式缓解：

- Stage 1 的 KL 正则让 latent 空间更平滑；
- 直接蒸馏 `z_hat`；
- 再监督 LaWM 输出的 `FT_hat`；
- Action Expert 在训练时就使用预测的 `FT_hat`；
- 每执行一个 action chunk 就重新观察和预测。

但它没有彻底解决多模态未来问题。论文没有采用多 latent 采样、best-of-N 或候选未来规划。

最准确的理解是：

```text
IDM 定义 latent action 的“语言”
VLM 学习说这种语言
LaWM 检查 VLM 说出的内容能否产生正确未来
```

所以两者不是天然一样，而是通过 Stage 2 的“latent 数值蒸馏 + 未来特征监督”被强行对齐。
````

## 消息 13：用户

````text
stage2在训练的时候，LaWM是冻住的吧
````

## 消息 14：回答

````markdown
不能简单说“完全冻住”。论文明确的是：

> Stage 2 中，Action Expert 的动作损失不能反向更新 LaWM；但论文没有明确声明 LaWM 对所有损失完全冻结。

梯度关系更可能是：

```text
L_distill
  → 更新 VLM / policy prior

L_wm = MSE(LaWM(F1, z_hat), FT)
  → 更新 policy prior
  → 可能继续微调 LaWM

L_action
  → 更新 Action Expert
  × 不允许经过 KI 破坏 LaWM
```

也就是：

```text
LaWM 对 L_action 是“冻结/stop-gradient”的
LaWM 对 L_wm 不一定冻结
```

支持这一判断的依据：

1. 论文只明确称 DINO 是 frozen encoder，没有明确称 Stage 2 的 LaWM frozen。
2. Stage 2 包含子目标损失：

```text
L_wm = MSE(FT_hat, FT)
FT_hat = LaWM(F1, z_hat)
```

3. 作者特意使用 Knowledge Insulation，表述为：

```text
防止 Action Expert 的梯度覆盖 LaWM 的预训练动力学
```

如果 LaWM 对所有损失都完全冻结，这个说法会有些多余。

4. 附录称 Action Expert 使用 `1e-4` 学习率，“其他模块”使用 `3e-5`，也倾向于说明部分上游模块会继续训练。

不过，全冻结在数学上也是可行的：

```text
固定 LaWM 参数
L_wm 经过 LaWM 的计算图
只更新产生 z_hat 的 VLM
```

此时 LaWM 相当于固定的可微判别器，强迫 VLM 输出能被它正确解码的 latent action。

因此，基于当前 arXiv v1，最严谨的结论是：

```text
确定：L_action 不更新 LaWM
很可能：L_wm 可以小学习率微调 LaWM
不确定：作者是否在实现中彻底冻结了 LaWM 参数
```

论文没有给出 optimizer 参数组或明确的 `requires_grad` 配置，所以无法仅凭正文断言“Stage 2 的 LaWM 完全冻住”。
````

## 消息 15：用户

````text
ok,把论文解读和我们讨论的内容写成md文档存在本地，把每个stage训练的loss写清楚每个部分具体是什么。还有实验部分和附录。
````

## 消息 16：回答

````markdown
已完成并打开：

[LaWAM-论文解读与讨论.md](C:/Users/r11379/Desktop/wm/wm_reward/LaWAM-论文解读与讨论.md)

内容包括：

- Stage 1/2 完整流程与各项 Loss
- IDM、latent action、LaWM、VLM 的关系
- Stage 2 冻结与梯度流争议
- Flow Matching 动作损失
- 混合控制频率与时间编码
- LIBERO、RoboTwin、真实机器人实验
- 全部附录解读
- 局限、复现缺口及建议实验

所有公式均使用纯文本，避免 Cursor 渲染问题。
````

## 消息 17：用户

````text
md文档里面写一下论文的时间和发表单位
````

## 消息 18：回答

````markdown
已补充：

- 首次公开时间：2026-06-14
- 发表平台及预印本状态
- 清华、吉林大学、南开、北大、哈工大等全部作者单位。
````

## 消息 19：用户

````text
我问的问题和你的回答要完整保存在md中
````

## 消息 20：回答

````markdown
已更新文档，在第 16 节按时间顺序完整保存了 10 组问题与回答：

[打开 LaWAM-论文解读与讨论.md](C:/Users/r11379/Desktop/wm/wm_reward/LaWAM-论文解读与讨论.md)

公式已统一转成纯文本；工具调用和内部记录未收录。
````

## 消息 21：用户

````text
听不懂什么是完整吗，不要你总结
````

## 消息 22：回答

````text
已改正。第 16 节现在保存的是逐字问答原文，不再改写、整理或总结；包括你每次的原始问题和我当时的完整最终回答。
````

## 消息 23：用户

````text
action expert的输入是什么
````

## 消息 24：回答

````markdown
Action Expert 主要接收四类输入：

1. **带噪动作块**

```text
a_t
```

- 训练时：真实动作块与高斯噪声插值得到 `a_t`；
- 推理时：从纯高斯噪声开始；
- 同时输入 flow-matching 的生成时间 `t`。

2. **VLM 语义上下文**

```text
H_VLM = VLM(主视角, 语言指令, 其他视角)
```

它告诉 Action Expert：

- 当前任务是什么；
- 要操作哪个物体；
- 主视角和辅助视角中的语义信息。

3. **LaWM 动力学上下文**

```text
u      = DINO(当前主视角)
uT_hat = LaWM(u, z_hat)

DynamicsContext = Concat(u, uT_hat)
```

其中：

- `u` 是当前 DINO 特征；
- `uT_hat` 是 LaWM 预测的未来视觉子目标；
- 二者经过 Knowledge Insulation 后组成 dynamics stream。

它告诉 Action Expert：

```text
场景现在是什么样
+
动作块结束时应该变成什么样
```

4. **动作 token 的物理时间编码**

```text
physical_time_i = i / control_frequency
```

该编码告诉模型每个动作 token 对应多少真实时间，用于处理不同机器人的混合控制频率。变长动作块还会带 padding mask。

因此可以把完整输入写成：

```text
ActionExpert(
    noised_action_chunk = a_t,
    flow_time = t,
    semantic_context = H_VLM,
    dynamics_context = Concat(u, uT_hat),
    physical_time_encoding,
    attention_mask
)
```

Action Expert 预测动作 flow/velocity，经过约 10 次积分或去噪后得到 EEF action chunk。

需要注意：

```text
z_hat 不直接输入 Action Expert
```

它通过下面的路径间接影响动作：

```text
z_hat
  → LaWM
  → uT_hat
  → Action Expert
```

此外，论文所有实验都不向 Action Expert 输入 proprioceptive state，只使用 RGB 视觉上下文。
````

## 消息 25：用户

````text
VLM 语义上下文是latent action吗
````

## 消息 26：回答

````markdown
不是。**VLM 语义上下文和 latent action 是两种不同的信息**，虽然它们都来自 VLM。

```text
当前图像 + 指令 + 其他视角
              ↓
             VLM
        ┌─────┴──────────┐
        ↓                ↓
VLM 语义上下文       LA Query hidden state
H_VLM                  ↓ aggregation
        │             z_hat
        │               ↓
        │             LaWM
        │               ↓
        │             uT_hat
        │               │
        └──────┬────────┘
               ↓
         Action Expert
```

### VLM 语义上下文 `H_VLM`

它是一整组 VLM hidden tokens，保留：

- 当前图像内容；
- 语言指令；
- 物体语义和空间关系；
- 其他相机视角信息。

它直接进入 Action Expert 的 Semantic Attention，回答：

```text
任务是什么？
操作哪个物体？
指令的语义约束是什么？
```

### Latent action `z_hat`

`z_hat` 由 LA Query 的 hidden state 经过 query aggregation/projection 得到：

```text
z_hat = Aggregate(LA_Query_hidden)
```

它是一个更紧凑的视觉转移代码，回答：

```text
下一段时间内，场景应该发生什么变化？
```

`z_hat` 不直接进入 Action Expert，而是先经过：

```text
uT_hat = LaWM(u, z_hat)
```

转换成空间化的未来视觉子目标。

因此 Action Expert 同时使用：

```text
H_VLM
  → 语义流：任务意图

Concat(u, uT_hat)
  → 动力学流：当前状态到未来子目标的变化
```

一句话区分：

```text
VLM 语义上下文：完整地说明“要做什么”

latent action：压缩地表示“下一步要发生什么变化”
```
````

## 消息 27：用户

````text
讲一下实验结果和消融实验
````

## 消息 28：回答

````markdown
先给结论：

> LaWAM 最有说服力的优势是推理效率，而不是大幅提高成功率。消融实验支持“latent visual subgoal 确实有用”，但还没有完全拆清 LaWM、蒸馏和 Knowledge Insulation 各自的独立贡献。

## 一、主实验结果

### 1. LIBERO

LIBERO 包含 4 个 suite、40 个任务，每个任务测试 50 次，共 2,000 次。

| 方法 | Long | Goal | Object | Spatial | 平均成功率 | 延迟/chunk |
|---|---:|---:|---:|---:|---:|---:|
| pi0.5 | 92.4 | 98.0 | 98.2 | 98.8 | 96.9 | 220 ms |
| VLA-JEPA | 95.8 | 97.2 | 99.6 | 96.2 | 97.2 | — |
| Fast-WAM | 95.2 | 97.0 | 100.0 | 98.2 | 97.6 | 486 ms |
| Cosmos-Policy | 97.6 | 98.2 | 100.0 | 98.1 | 98.5 | 1,413 ms |
| LingBot-VA | 98.5 | 97.2 | 99.6 | 98.5 | 98.5 | 4,482 ms |
| **LaWAM** | **97.0** | **98.4** | **99.6** | **99.4** | **98.6** | **187 ms** |

LaWAM 的表现：

- 平均成功率 98.6%，表中最高；
- Long 任务 97.0%；
- Goal 任务 98.4%；
- Object 任务 99.6%；
- Spatial 任务 99.4%；
- 完整 action chunk 推理延迟为 187 ms。

延迟方面：

```text
相对 pi0.5：约快 1.18 倍
相对 Fast-WAM：约快 2.6 倍
相对 Cosmos-Policy：约快 7.6 倍
相对 LingBot-VA：约快 24 倍
```

但成功率需要保守理解：

```text
LaWAM：98.6%
次优方法：98.5%
差距：0.1 个百分点
```

在 2,000 次 trial 中，0.1 个百分点约等于 2 次成功的数量级。论文没有报告多训练种子、方差或置信区间，所以不能证明这 0.1 个百分点具有统计显著性。

因此 LIBERO 的准确结论是：

```text
LaWAM 达到了最强一档成功率，
同时显著降低了 pixel-space WAM 的推理延迟。
```

而不是：

```text
LaWAM 的任务成功率显著超过所有方法。
```

### 2. RoboTwin 2.0

RoboTwin 包含 50 个双臂操作任务，每个任务分别在 clean 和 randomized 场景测试 100 次。

| 方法 | Clean SR | Randomized SR | 两者平均 |
|---|---:|---:|---:|
| Fast-WAM | 91.98 | 90.52 | 91.25 |
| GigaWorld-Policy | 86.36 | 85.04 | 85.70 |
| LingBot-VA | 91.50 | **90.92** | 91.21 |
| pi0.5 | 82.74 | 76.76 | 79.75 |
| Motus | 88.66 | 87.02 | 87.84 |
| **LaWAM** | **92.64** | 89.80 | **91.22** |

结果说明：

- LaWAM 在 clean setting 中第一，达到 92.64%；
- 比 Fast-WAM 高 0.66 个百分点；
- randomized setting 为 89.80%，不是第一；
- 比 LingBot-VA 低 1.12 个百分点；
- 比 Fast-WAM 低 0.72 个百分点；
- clean/random 合并平均约 91.22%，与最强 pixel WAM 基本处于同一水平。

因此 RoboTwin 支持的是：

```text
单步 latent subgoal 能扩展到复杂双臂操作，
并在不生成未来视频的情况下保持竞争力。
```

它不支持“所有随机化场景都优于 pixel WAM”。

LaWAM 在部分任务上仍明显较弱：

| 任务 | Clean | Randomized |
|---|---:|---:|
| Hanging Mug | 51 | 43 |
| Open Microwave | 41 | 43 |
| Turn Switch | 47 | 56 |
| Place Can Basket | 92 | 65 |
| Stack Blocks Three | 90 | 75 |

这些任务通常需要更精细的几何、接触或多阶段控制。

### 3. 真实机器人实验

每个任务测试 30 次。

| 方法 | Pick-and-Place | Open Drawer | Fold Towel | 平均 |
|---|---:|---:|---:|---:|
| pi0.5 | 86.7 | 80.0 | 83.3 | 83.3 |
| GR00T-N1.6 | 83.3 | 76.7 | 46.7 | 68.9 |
| Fast-WAM | 56.7 | 63.3 | 70.0 | 63.3 |
| LingBot-VA | 76.7 | 83.3 | 0.0 | 53.3 |
| **LaWAM** | **93.3** | **86.7** | **90.0** | **90.0** |

LaWAM 在三个任务上都是第一：

- Pick-and-Place：93.3%；
- Open Drawer：86.7%；
- Fold Towel：90.0%；
- 平均：90.0%。

毛巾折叠最能体现低延迟的实际价值。毛巾在模型推理期间仍会继续运动，高延迟模型输出动作时，动作可能已经对应旧状态：

```text
采集观测
  ↓ 模型推理耗时过长
毛巾状态已经变化
  ↓
动作与当前状态不匹配
```

不过每任务只有 30 次，成功率的最小变化单位是 3.33 个百分点。LaWAM 90.0% 对 pi0.5 83.3%，总计约多成功 6/90 次，仍然需要更多试验和置信区间。

### 4. LaWM 是否真的学到了 dynamics

论文使用两类证据。

#### 500 条 LIBERO open-loop rollout

作者比较：

```text
LaWM rollout 与真实未来的特征相似度
真实未来与初始状态的特征相似度
LaWM rollout 与初始状态的特征相似度
```

结果显示：

- rollout 与真实未来保持较高相似度；
- rollout 会逐渐远离初始特征；
- LaWM 不是简单复制当前状态。

#### 跨环境、跨具身 rollout

将同一 latent-action sequence 应用到不同初始场景：

```text
相同 latent action
+ 不同初始视觉状态
→ 不同但上下文合理的 latent rollout
```

这支持：

- latent action 表示相对抽象的视觉转移；
- LaWM 将该转移绑定到当前场景和机器人。

但这些主要是 latent 可视化，不是未见机器人上的真实闭环执行成功率，也没有验证多个候选真实动作的反事实后果。

## 二、组件消融实验

Figure 6 比较了以下配置：

```text
w/o WM
w/o KI & distill
w/o distill
w/o pretrain
完整 LaWAM
```

论文没有提供完整数值表，只给出柱状图和定性结论，因此不应该凭图猜测精确数字。

### 1. `w/o WM`

该配置移除 LaWM 提供的 latent visual subgoal。

结果：

- 性能下降最大；
- LIBERO-Long 上下降尤其明显。

说明只使用 VLM 或紧凑 latent-action 信息不够，Action Expert 从空间化的未来特征中获得了明显收益。

作者据此认为：

```text
latent action 只有先经过 LaWM，
展开成当前场景中的未来视觉特征，
才最适合指导动作生成。
```

但该消融还不能完全排除：

- LaWM 带来的额外参数量；
- 额外的 future-feature supervision；
- 更深的网络计算；

也是性能提升的一部分。

### 2. `w/o distill`

移除：

```text
L_distill = MSE(z_hat, z_teacher)
```

此时 VLM 不再直接模仿 IDM 定义的 latent action，只能通过 subgoal loss 和 action loss间接学习如何驱动 LaWM。

结果明显下降，说明：

- LaWM 对输入 latent 的分布比较敏感；
- 只靠终点特征误差不足以稳定学习接口；
- VLM 需要明确学习 IDM 的 latent 坐标系。

### 3. `w/o KI & distill`

该配置同时去掉：

- Knowledge Insulation；
- latent-action distillation。

性能比仅去掉 distillation 进一步下降。

作者的解释是：

```text
一方面，z_hat 没有与 IDM latent 对齐；
另一方面，Action Expert 的梯度可能破坏预训练动力学表示。
```

但这里存在一个实验设计限制：

```text
论文没有单独报告只去掉 KI 的完整结果。
```

因此无法从该组合消融中精确计算 Knowledge Insulation 自身贡献了多少。

### 4. `w/o pretrain`

去掉大规模预训练后性能下降，说明：

- 4,500 小时视频学到的 dynamics prior 有价值；
- 仅依靠 benchmark 后训练数据不足以达到完整模型性能。

但论文没有进一步分别消融：

```text
去掉人类视频
去掉机器人无动作标签视频
改变各数据源比例
```

所以无法知道 1,500 小时人类视频单独贡献了多少。

### 5. 完整 LaWAM

完整模型同时包含：

```text
LaWM 预训练
+ latent-action distillation
+ subgoal supervision
+ Knowledge Insulation
+ Action flow matching
```

在四个 LIBERO suite 上整体最好。

## 三、混合控制频率消融

Appendix Figure 7 比较：

```text
w/o pos
w/ pos
only 20Hz
```

作者从同一批原生 20 Hz LIBERO 轨迹下采样得到 10 Hz 和 5 Hz 数据，再联合训练 5/10/20 Hz。

### `w/o pos`

混合三种频率，但不加入物理时间编码。

结果明显下降，因为相同 action token 下标对应不同真实时间：

```text
5 Hz 的第 5 个 token  = 1.0 秒
20 Hz 的第 5 个 token = 0.25 秒
```

### `w/ pos`

加入：

```text
physical_time_i = i / control_frequency
```

性能大部分恢复到原生 20 Hz 模型附近。

### `only 20Hz`

只使用原始 20 Hz 数据，是该受控实验的上界参考。

这项消融比 Figure 6 更干净，因为：

- 使用相同原始轨迹；
- 任务、视觉域、具身和语言完全相同；
- 只改变控制频率与时间编码。

因此它较有力地证明了物理时间编码能够缓解 mixed-frequency confusion。

## 四、综合判断

实验最强地支持：

```text
1. latent-space future prediction 比视频生成快很多；
2. latent visual subgoal 对动作生成确实有帮助；
3. latent-action distillation 对稳定驱动 LaWM 很重要；
4. 物理时间编码对混合频率训练有效。
```

实验尚未充分证明：

```text
1. LaWM 能进行多候选反事实规划；
2. Knowledge Insulation 的独立贡献有多大；
3. 人类视频的独立贡献有多大；
4. 0.1 个百分点的 LIBERO 领先具有统计显著性；
5. 跨具身 latent rollout 等价于真实跨具身控制迁移。
```

所以这篇论文最可靠的结论是：

> LaWAM 用一个轻量、单次前向的 latent subgoal，获得了接近或达到最强 WAM 的控制性能，同时显著降低了推理延迟。
````
