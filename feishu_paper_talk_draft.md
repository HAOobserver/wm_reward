<callout emoji="🧭" background-color="light-blue">这份文档按“基本信息 → 原论文框架图 → 方法与训练 → 实验 → 附录要点 → 讨论中的关键质疑”组织。重点不是复述摘要，而是解释每个信号到底来自哪里、更新谁、是否真的属于 RL、实验能证明到什么程度。</callout>

## 一、先建立共同问题：四篇论文都在补 SFT 的哪块短板

离线行为克隆/SFT 只在专家访问过的状态上学习动作：训练时状态来自专家分布，部署时状态来自模型自己的闭环行为。一次小误差可能让机器人进入专家数据没覆盖的状态，后续误差继续累积。

四篇论文都在处理这个问题，但路线不同：

| 论文 | 核心补充信号 | 最直观的定位 |
|---|---|---|
| VLA-RFT | 世界模型预测的动作后果与专家参考后果的视觉相似度 | 世界模型辅助、专家锚定的 RL 后训练 |
| WAM-RL | 世界模型“想象视频”与环境“执行视频”的重建一致性 | Actor RL + 世界模型在线视频 SFT |
| LaWAM | 动作块结束时的未来 DINO 特征子目标 | 非 RL 的 latent-future 条件化 VLA |
| VLA-OPD | Teacher 在学生自己访问的状态上给出的动作分布 | On-policy 交互式策略蒸馏 |

<callout emoji="💡" background-color="light-green">贯穿全文的判断标准：不要只看作者是否用了“RL”“world model”或“on-policy”这些词，而要连续追问四件事：训练状态由谁产生？信号来自哪里？梯度更新哪个模块？这个信号是否真的等价于任务成功？</callout>

## 二、时间与第一单位速览

| 论文 | 首次公开时间 | 版本/状态（以本文所读版本为准） | 第一署名单位 |
|---|---|---|---|
| VLA-RFT | 2025-10-01 | arXiv:2510.00406v1，预印本；未据本文确认正式会议或期刊录用 | Westlake University（西湖大学） |
| WAM-RL | 2026-06-16 | arXiv:2606.17906v1，cs.RO；本文以本地 arXiv 稿为准 | 北京大学计算机学院、多媒体信息处理全国重点实验室 |
| LaWAM | 2026-06-14 | arXiv:2606.15768v1，cs.RO，预印本 | 单位列表第一项为清华大学；第一作者 Jialei Chen 的署名单位为吉林大学、中关村学院与 Striding.AI |
| VLA-OPD | 2026-03-27 | arXiv:2603.26666v1，cs.RO，预印本 | 香港科技大学（广州） |

> “第一单位”可能指“单位列表第一个”或“第一作者的第一单位”。LaWAM 两者不同，因此这里分别写明，汇报时不要含混。

---

## 三、VLA-RFT：在世界模型里做专家附近的后果搜索

### 3.1 基本信息与一句话结论

- 全名：**VLA-RFT: Vision-Language-Action Reinforcement Fine-Tuning with Verified Rewards in World Simulators**
- 首次公开：2025-10-01，arXiv:2510.00406v1。
- 第一署名单位：西湖大学；其余单位包括浙江大学、OpenHelix Team、复旦大学、郑州大学、北京邮电大学、河北工业大学。
- 基础策略：VLA-Adapter，上层是 VLM，下层是基于 DiT/Flow Matching 的连续动作头。
- 附录所列骨干包括 DINO/SigLIP 视觉编码与 Qwen2.5-0.5B 语言骨干；这些属于 LIBERO 实验所用轻量配置。

<callout emoji="🎯" background-color="light-blue">一句话：当前 VLA 从同一个初始状态采样多组动作，冻结的动作条件世界模型预测每组动作的未来视觉轨迹，再用“候选轨迹与专家参考轨迹有多像”作为奖励，通过 GRPO 更新动作生成模块。</callout>

### 3.2 原论文框架图

<image url="https://i0.wp.com/vla-rft.github.io/static/images/Figure1.png?resize=1200%2C628&ssl=1" width="1000" align="center" caption="VLA-RFT 原论文 Figure 1：用世界模型承接多组 VLA 动作 rollout，并以轨迹奖励进行 GRPO 后训练。"/>

### 3.3 如何逐块读图

1. **起点**：离线数据给出当前图像、语言指令、机器人状态，以及一条专家动作/视觉参考轨迹。
2. **VLA policy**：不是只输出一个动作，而是从同一起点随机采样多组 action chunks；论文配置中每个起点采样 16 组。
3. **World Model**：每组动作被送入冻结的动作条件视频世界模型，产生对应的未来图像序列。
4. **Reward**：比较候选动作造成的预测视觉后果与专家动作造成的参考后果。
5. **GRPO**：在同一组 16 条候选中做相对比较；高于组平均的候选提高概率，低于组平均的候选降低概率。
6. **部署**：世界模型只用于训练。部署时保留微调后的 VLA，不需要在线生成视频或计算奖励。

图中“end-to-end update VLA”的表达需要谨慎：附录说明 RFT 阶段冻结上层 VLM 和世界模型，实际更新的是 Flow action head 与新增的 Sigma Net。因此严格说是**动作生成模块的强化微调**，不是整个 VLA 全参数端到端 RL。

### 3.4 Stage I：世界模型和 VLA 的预训练

#### 世界模型

世界模型接收初始画面和 7 维机器人动作序列，自回归预测未来画面。实现上：

- 图像先经 VQGAN 类 tokenizer 变为离散 image tokens；
- 连续动作变为 action tokens；
- 12 层、hidden size 768、约 138M 参数的 Transformer 预测未来 image tokens；
- decoder 再恢复为未来 RGB 帧。

其训练目标是下一帧/后续帧最大似然。因为是自回归 rollout，短期可以连贯，但预测误差会随时间累积。

附录报告其在 LIBERO 离线片段上的指标：MSE 0.0039、PSNR 25.23 dB、SSIM 0.906、LPIPS 0.059。它们能证明数据分布内短期重建较好，却不能证明世界模型对当前策略提出的 OOD 动作仍然准确。

#### VLA 的 Flow Matching 预训练

对专家动作块 `a` 采样高斯噪声 `ε`，构造噪声动作 `a^τ = τa + (1−τ)ε`，并让动作头预测从噪声指向专家动作的速度 `a−ε`。推理时从高斯噪声开始，沿学到的向量场积分得到连续动作块。

这一步仍是标准模仿学习。它提供可靠起点：如果直接从随机策略做 RL，动作会迅速进入世界模型最不可信的区域。

### 3.5 为什么把 Flow ODE 改成 SDE policy

原始 Flow ODE 在给定初始噪声后，中间去噪转移是确定性的；GRPO/PPO 需要随机探索，也需要计算新旧策略对同一采样轨迹的概率比。

作者新增 **Sigma Net**：

- Flow head 给出去噪下一步的均值方向；
- Sigma Net 给出每个去噪步骤的方差；
- 下一步从高斯分布中采样；
- 因而每一步都有可计算的 log probability。

论文使用 10 个去噪步、步长 0.1，对各去噪转移的 log probability **取平均**，再构造新旧策略概率比。Sigma Net 的主要作用不是增强视觉理解，而是把 Flow 动作生成器变成**可采样、可计算概率、可用策略梯度更新**的随机策略。

### 3.6 一次 RFT 更新的完整数据流

```text
离线样本中的初始图像、语言、机器人状态
                 ↓
当前 VLA / SDE action head 采样 16 个动作块
                 ↓
冻结世界模型分别预测 16 条未来视觉轨迹
                 ↓
候选动作与专家动作均经同一冻结 World Model 生成轨迹
再计算两条生成轨迹的 L1 + LPIPS 相似度奖励（最终 R3）
                 ↓
同一起点内计算相对 advantage
                 ↓
GRPO 更新 Flow action head + Sigma Net
```

重要边界：论文主要是“一次生成完整 action chunk，再由世界模型展开这个动作块”。它不是每预测一帧就把新画面反馈给 VLA 重新规划，因此更接近**短期动作块后果预测**，不是完整的长时闭环 MPC。

### 3.7 三种奖励与最终方案

1. **R1：动作距离**  
   `R1 = −||a_policy − a_expert||_1`。这是最直接的动作模仿，无法容纳“动作不同但结果同样正确”的多解性。成功率 87.7%。

2. **R2：生成轨迹 vs 真实专家帧**  
   候选动作经过世界模型得到预测帧，再与数据集/仿真器中的真实专家帧比较 L1 和 LPIPS。问题是比较两侧一个是生成图像、一个是真实渲染图像，奖励会混入世界模型的画质偏差。成功率 87.1%。

3. **R3：同一世界模型空间内比较**  
   候选动作和专家动作都通过同一个世界模型，再比较两条生成轨迹。双方共享相似的模糊、纹理与颜色偏差，可以抵消一部分静态生成偏差。成功率 91.1%。

<callout emoji="⚠️" background-color="light-yellow">论文正文公式 11 的写法更像 R2，而消融章节将最终方案描述成 R3；两处并不完全一致。复现时应以代码或作者说明确认，不能把文中歧义悄悄抹平。</callout>

相对 86.6% Base，R1/R2/R3 分别约为 `+1.1 / +0.5 / +4.5` 个百分点；真正拉开差距的是在同一生成空间比较的 R3，而不是简单把动作 L1 换成图像 L1。

### 3.8 GRPO 目标与训练配置

同一起点的 16 条轨迹先计算组平均奖励，单条 advantage 为“自身奖励减组平均”。所以即使所有视觉距离奖励都是负数，仍然可以通过相对优劣更新。

例如组平均为 −0.14 时，奖励 −0.10 的候选仍有正 advantage，−0.18 的候选则有负 advantage。

最终损失包含三部分：

- GRPO/PPO-style clipped policy objective：提高组内较优候选的概率；
- 0.01 倍的 Flow Matching MSE：继续把动作头约束在专家附近，防止 RL 漂移；
- 0.003 倍的熵奖励：避免 Sigma Net 过早把方差压到零。

关键配置：Flow head 学习率 `1e−6`，Sigma Net 学习率 `1e−5`，batch size 16，每个起点 16 个 rollout，RFT 400 个更新步；VLA 和世界模型在此之前各自已训练 150K 步。

“只训练 400 步”不能理解为只看了 400 条轨迹。按配置粗算，模型内 rollout 数量可达 `400 × 16 × 16 = 102,400`，每条还包含 10 个 SDE 去噪步骤和多帧世界模型生成。论文也没有给出统一 wall-clock、FLOPs 或能耗，因此不能直接说它比 150K SFT 便宜 375 倍。

论文横向比较表还混合了作者复现值与其他论文报告值；不同方法的 batch、rollout 和环境成本并不统一，不能只按 optimizer step 宣称便宜几十倍。

### 3.9 实验结果

#### 标准 LIBERO

| 方法 | Spatial | Object | Goal | Long | 平均 |
|---|---:|---:|---:|---:|---:|
| Base SFT | 88.4 | 88.0 | 92.8 | 77.2 | 86.6 |
| VLA-RFT | 94.4 | 94.4 | 95.4 | 80.2 | 91.1 |

平均提升 4.5 个百分点。正文一处写 91.3% / +4.7，但表 2、表 4、表 9、表 10 均为 91.1% / +4.5，汇报应采用多处一致的 91.1%。

#### 扰动环境

作者扰动物体位置、目标位置、机器人初始状态及其组合。8 种轻/重设置中都优于 Base，提升约 2.5–6.7 个百分点。但重度组合扰动仅从 34.0% 到 37.0%：说明“相对更鲁棒”成立，“已经有很强 OOD 鲁棒性”不成立。

#### 证据边界

- 只有 LIBERO 仿真，没有真实机器人；
- 没有多随机种子、标准差或置信区间；
- 世界模型指标主要来自数据分布内动作；
- 未报告模型奖励与真实成功率的相关性；
- 未检查 policy 是否利用世界模型漏洞。

### 3.10 附录中最有意义的点

1. **实际冻结范围**：RFT 冻结 VLM 和世界模型，只更新动作头与 Sigma Net，修正了正文“端到端更新”的过强表述。
2. **世界模型细节**：138M、12 层、hidden 768、训练 150K、片段长度 8，以及四项重建指标。
3. **训练预算细节**：16 rollouts、batch 16、400 updates、两个学习率与熵/MSE 系数，说明 400 step 背后并不轻。
4. **完整消融表**：R3 显著优于 R1/R2，说明“同一生成空间比较”是主要有效设计。
5. **完整扰动表**：能看到某些设置只是小幅改善，避免只读平均数得出过强结论。

### 3.11 讨论重点：它到底是不是 RL，有没有探索

**从优化形式看，是 RL。** 当前策略随机采样动作，世界模型产生后果，奖励变成 advantage，再通过 `∇ log π(a|s) × A` 更新，梯度不是普通专家动作回归。

**从监督目标看，又是轨迹模仿。** 奖励完全锚定专家视觉轨迹，不是独立的任务成功、物理约束或程序验证。最准确的定位是：

> 专家锚定的模型式强化后训练，或者“使用 RL 优化器的轨迹模仿”。

**它有探索，但主要是局部探索。** 探索来自初始高斯噪声、Sigma Net、16 个候选和熵正则；然而逐帧 L1+LPIPS、辅助 MSE、PPO clipping、SFT 初始化和世界模型 OOD 风险都把策略限制在专家附近。不同速度、不同路径、恢复重试甚至更高效的成功策略，都可能因中间视觉轨迹不像专家而被扣分。

它仍可能优于 SFT，是因为训练信号从“动作数值是否像专家”转向“动作后果是否像成功参考”，并在同一离线起点上评估多组当前策略动作，相当于在专家邻域做了一次后果驱动的局部数据增强。

因此不要说“完全没有探索”，更准确的说法是：

> 它在专家分布附近做受约束的随机搜索，寻找能更稳定复现专家视觉后果的动作，而不是开放式发现新策略。

### 3.12 汇报收束

VLA-RFT 最有价值的思想是：**世界模型不一定在部署时做规划，也可以只在训练阶段充当动作后果模拟器。** 但其所谓 verified reward 只是自动可计算的代理奖励，并非真实任务验证；当前证据更支持“强力的世界模型辅助 VLA 后训练概念验证”，不支持“已经解决真实机器人开放式强化学习”。

<!-- CHUNK -->

## 四、WAM-RL：让“现实接近想象”，再让“想象学习成功现实”

### 4.1 基本信息与一句话结论

- 全名：**WAM-RL: World-Action Model Reinforcement Learning with Reconstruction Rewards and Online Video SFT**
- 首次公开：2026-06-16，arXiv:2606.17906v1，cs.RO。
- 第一署名单位：北京大学计算机学院、多媒体信息处理全国重点实验室；其他署名单位为 Northeastern University、清华大学。
- Xiaowei Chi 标注为 Project Leader，Shanghang Zhang 为通讯作者。
- 当前为 8 页 arXiv 预印本，正文未标注正式会议或期刊录用。
- 基座为 Genie Envisioner-ACT；论文仅给出联合在线训练使用 8×A800、约 8 小时，未给完整环境帧数与模块级预算。

<callout emoji="🎯" background-color="light-blue">一句话：World Model 先想象未来视频，Actor 把 latent 计划翻译成动作并在环境中执行；预测视频与执行视频的相似度用于 RL 更新 Actor，而最终成功的执行视频用于继续 SFT 世界模型。</callout>

### 4.2 原论文框架图

<image url="https://arxiv.org/html/2606.17906v1/x1.png" width="1000" align="center" caption="WAM-RL 原论文 Figure 1：World Model 与 Actor 通过在线环境交互形成两条协同更新路径。"/>

*图中写有 “Real World Execution”，但本文实验实际都在 LIBERO/RLBench 仿真中完成；真实机器人部署只是可能的扩展，并未被本文验证。*

### 4.3 先理解 World-Action Model 的角色分工

普通 VLA 直接从当前观察和指令预测动作。World-Action Model 则显式生成未来视觉/latent：

```text
当前观测、任务信息
        ↓
World Model 想象未来视频并产生 latent
        ↓
Actor 把 latent 计划翻译为动作
        ↓
环境执行，返回真实视频与 success/fail
```

论文的核心假设是：

- World Model 更像“导演/规划器”，决定未来应该发生什么；
- Actor 更像“演员/翻译器”，决定怎样把 latent 计划变为可执行动作。

只优化 Actor 能提高翻译精度，却无法修复长时程中上游 World Model 想错的问题，所以作者提出两条协同路径。

### 4.4 图中的两条训练路径

#### 路径 A：Actor 的强化学习

World Model 先预测未来 `H` 帧 `x_hat`，Actor 执行动作后，环境给出真实未来 `H` 帧 `x`。奖励定义为两段视频的相似度：

`r_t = sim(x_hat_(t+1:t+H), x_(t+1:t+H))`

论文测试 Pixel MSE、Optical Flow MSE、DINOv2 feature MSE、V-JEPA2 feature distance。效果最好的 Pixel MSE 可直观还原为 `reward ≈ −MSE(预测帧, 执行帧)`。

该奖励对成功和失败 rollout 都会计算 advantage，并通过 Flow-SDE policy gradient 更新 **Actor**。它不直接更新 World Model；success/fail 只负责筛选哪部分视频进入下一条 World Model 更新路径。

#### 路径 B：World Model 的 Online Video SFT

整条在线 rollout 若最终成功，就把真实执行视频拿来继续训练 World Model 的未来视频预测。因为未来帧本身就是监督标签，不需要人工逐帧标注，作者称其为 self-supervised video fine-tuning。

成功轨迹不一定每一步都成功。例如：

```text
第一次抓偏
   ↓
调整夹爪
   ↓
重新抓取
   ↓
最终任务成功
```

因此成功 rollout 可能把恢复与重试行为带进 World Model。

反面是 success-only 选择偏差：早期成功率低时可用于 Video SFT 的数据很少，而失败轨迹中本可用于学习动力学的信息被丢弃。

### 4.5 Actor 为什么使用 Flow-SDE

Actor 原本使用 Flow Matching ODE 生成动作。论文在每个去噪转移中加入布朗噪声，形成 Flow-SDE：

- 相同初始噪声下，中间转移不再完全确定；
- 每个转移是可计算密度的高斯分布；
- 完整动作生成轨迹的 log probability 可以写成各步 log probability 之和；
- 高奖励动作提高对应内部去噪转移的概率，低奖励动作降低概率。

严格说，原始 Flow ODE 从不同初始噪声出发也能产生不同动作，所以不能说“ODE 完全没有随机性”。准确表述是：**给定初始噪声后，ODE 的中间转移确定；Flow-SDE 在每个转移中显式加入可计算概率的随机性。**

### 4.6 Reconstruction Reward 到底比较哪两段视频

两段视频分别是：

1. World Model 根据当前观测预测的未来视频；
2. Actor 动作在 LIBERO/RLBench 仿真环境执行后，由仿真器渲染的 ground-truth RGB 视频。

“真实帧”不是数据集里预先保存的专家帧，也不是物理机器人摄像头画面，而是本轮在线动作实际执行后的仿真器观测。

四类相似度：

- **Pixel MSE**：逐像素比较 RGB；最终任务成功率最好。
- **Optical Flow MSE**：分别计算两段视频的二维像素运动场，再比较运动方向和幅度；不是机器人关节速度。
- **DINOv2 MSE**：比较图像语义特征。
- **V-JEPA2**：比较视频时空特征。

论文没有明确给出负号、缩放、裁剪、归一化、逐帧聚合、帧率同步和时间对齐细节，因此 `−MSE` 只是依据文字作出的合理还原，不是完整复现公式。

### 4.7 World Model 更新为何需要 latent KL

Actor 读取 World Model 的中间 latent。如果 World Model 在线更新过快，其“内部语言”发生漂移，Actor 原来学到的 latent-to-action 映射会失效。

作者保留一个冻结的旧 World Model，并把确定性 DiT latent 人为近似为高斯分布：

- 当前 latent 作为当前高斯均值；
- 旧 latent 作为参考高斯均值；
- 当前协方差用 EMA feature statistics 估计；
- 旧协方差固定；
- 对两者施加 KL 正则。

最终 World Model 损失为：

`L_WM = L_video + λ_KL L_KL`

直观上，Video SFT 允许“导演”学习新分镜，KL 要求它不要突然换一套 Actor 听不懂的语言。

这是工程化近似，不是标准概率 World Model。论文没有说明 KL 加在哪些 DiT 层、token/时间/batch 维如何统计、EMA 参数和 `λ_KL` 具体取值，也没有与简单 feature L2/cosine 对照。

还要区分“latent 接口稳定”与“reward 稳定”：World Model 在线更新后，预测视频本身会变化，Actor 面对的 reconstruction reward 目标也是非平稳的；latent KL 并不能保证 reward 数值不漂移。

### 4.8 按正文还原的完整训练循环

```text
1. World Model 根据当前观测生成未来视频和 latent
2. Actor 根据 latent，通过 Flow-SDE 采样动作
3. 在 LIBERO/RLBench 中执行动作
4. 收集真实执行视频和 success/fail
5. 预测视频 vs 执行视频 → reconstruction reward
6. policy gradient 更新 Actor
7. 若整条 rollout 最终成功：
      用执行视频做 Online Video SFT
      用 latent KL 约束 World Model
8. 使用更新后的两个模块继续交互
```

“联合优化”不是同一个 RL loss 同时更新两个模块：只有 Actor 接受 policy gradient，World Model 仍然通过视频预测损失更新。

### 4.9 实验结果

#### 主结果

| 方法 | LIBERO-Object | RLBench Water Plants |
|---|---:|---:|
| Base | 68 | 19 |
| πRL（Actor-only） | 78 | 18 |
| WAM-RL | 82 | 22 |

方向上支持作者观点：

- 短时程 LIBERO-Object 中，单独优化 Actor 已有显著收益；
- 长时程 Water Plants 中，Actor-only 没超过 Base；
- 联合更新达到 22%，比 Base 高 3 个百分点。

但长时程结论主要来自一个任务，22% 的绝对成功率仍很低，也没有随机种子、误差条、置信区间和评测回合数。

此外，Base、πRL 与 WAM-RL 同时改变了 reward 设计和是否更新 World Model，主表不是干净的 2×2 组件消融；因此只能支持方向性判断，不能把 4 个百分点全部归因于 Online Video SFT。

#### Reward 消融（Water Plants）

| Actor reward | 成功率 |
|---|---:|
| Base | 19 |
| πRL | 18 |
| Pixel MSE | 21 |
| Optical Flow MSE | 19 |
| DINOv2 MSE | 16 |
| V-JEPA2 | 17 |

Optical Flow 对成功/失败奖励分布的区分度更强，却没有带来最好策略；Pixel MSE 区分度较弱但最终成功率最高。说明“奖励能区分成功失败”与“奖励容易被策略稳定优化”不是同一件事。

主表 WAM-RL 为 22%，消融中 Pixel MSE 为 21%，论文没有解释差异，汇报时应指出。

### 4.10 附录/补充信息中真正值得注意的点

本地 arXiv v1 篇幅很短，缺少可无歧义复现的完整附录。最有价值的额外信息反而是它暴露出的实现缺口：

1. Flow-SDE 的噪声日程、去噪步数未完整给出；
2. advantage 方法、更新频率、Actor/World Model 先后顺序未给出；
3. batch size、optimizer、学习率、环境交互总量未给出；
4. reconstruction reward 的符号、归一化与时间对齐未给出；
5. latent KL 的层位置、统计维度、EMA 与权重未给出；
6. Figure 3 有 with/without Video SFT 的恢复行为定性对照，但没有“保持 Actor reward 完全相同、只切换 Video SFT”的量化 2×2 消融；
7. 没有无 KL 的量化对照。

### 4.11 讨论重点：Reward 与任务成功是否对齐

这是这篇最重要的问题。Actor reward 衡量：

> 真实执行是否符合 World Model 的想象。

它不直接衡量：

> 任务是否完成。

所以 `reconstruction reward 高` 不推出 `任务成功`。如果 World Model 预测“没有抓住杯子”，Actor 也确实没抓住，两段视频可能很像，奖励却很高。Pixel MSE 还可能被大面积静态背景主导。

反过来也可能“任务成功但 reward 低”：Actor 采用不同但有效的抓取路线、速度或中间轨迹时，Pixel MSE 会惩罚这种替代解。失败 rollout 仍会通过 reconstruction reward 更新 Actor，只是不会进入 World Model 的 success-only Video SFT。

任务成功并非完全缺席：基座 World Model 预训练于专家轨迹，也可能读取任务上下文；在线阶段又只把最终成功 rollout 用于 Video SFT。最明确的在线链路是：

```text
success/fail
   ↓ 只筛选最终成功视频
Online Video SFT 更新 World Model
   ↓
World Model 更倾向想象成功过程
   ↓
Actor reward 要求执行符合该想象
```

这是一条间接且延迟的任务对齐路径。Flow-SDE 虽然在动作空间中加入随机探索，但 reconstruction reward 会强烈惩罚与 World Model 视觉计划不同的路径，因此有效探索更像“在给定视觉计划附近寻找实现动作”，不是自由发现任意成功策略。

更稳妥的奖励应显式结合：

`r_total = λ_success r_success + λ_progress r_progress + λ_recon r_reconstruction`

其中任务奖励决定“做成什么”，重建奖励只负责“怎样稳定实现计划”。

### 4.12 汇报收束

WAM-RL 的核心不是“RL 让 World Model 画面更准”，而是：

> Actor RL 让现实接近想象；Online Video SFT 让想象接近成功的现实。

它抓住了 WAM 特有的模块耦合问题，但“执行一致性不等于任务成功”是方法的核心风险。当前实验更适合支持“有启发性的联合在线后训练概念验证”，不足以证明已经形成稳定、通用的长时程机器人 RL。

<!-- CHUNK -->

## 五、LaWAM：不生成 RGB 未来，只预测一个未来 latent 子目标

### 5.1 基本信息与一句话结论

- 全名：**LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies**
- 首次公开：2026-06-14，arXiv:2606.15768v1，cs.RO，23 页。
- 单位列表第一项为清华大学；第一作者 Jialei Chen 的署名单位为吉林大学、中关村学院和 Striding.AI。其他单位包括南开大学、北京大学、哈尔滨工业大学、无问芯穹。
- 完整模型约 2.3B，核心 LaWM 约 230M。

<callout emoji="🎯" background-color="light-blue">一句话：LaWAM 不迭代生成未来 RGB 视频，而是预测动作块结束时的 DINO 未来特征图，把它作为 latent visual subgoal，再让 Action Expert 根据当前特征与未来特征生成连续动作块。</callout>

### 5.2 原论文框架图

<image url="https://cdn.jsdelivr.net/gh/RLinf/LaWAM@main/assets/lawam_overview.png" width="1000" align="center" caption="LaWAM 原论文 Figure 2/官方 overview：Stage 1 学 latent-action-conditioned LaWM；Stage 2 让 VLM 驱动 LaWM 并生成动作。"/>

### 5.3 图的总逻辑

```text
Stage 1：
当前图像 o1 + 真实未来图像 oT
        ↓ 冻结 DINO
当前特征 F1 + 未来特征 FT
        ↓ IDM
latent action z
        ↓ LaWM(F1, z)
重建未来特征 FT

Stage 2：
当前图像 + 指令
        ↓ VLM / LA Query
预测 latent action z_hat
        ↓ LaWM(F1, z_hat)
未来 latent subgoal FT_hat
        ↓ Action Expert
连续 action chunk
```

Stage 1 定义 latent action 的“语言”；Stage 2 教策略在看不到未来时说这种语言。

### 5.4 Stage 1：Latent World Model 如何训练

#### 冻结 DINO 提取当前与未来特征

从视频中取当前观测 `o1` 和固定物理时间后的 horizon observation `oT`，用冻结的 distilled DINOv3 ViT-B/16 得到带空间布局的 patch feature map `F1` 和 `FT`。

LaWAM 认为控制不需要重建背景纹理、光照和所有 RGB 细节，只需要保留“机械臂/物体下一阶段应该移动到哪里”的空间化未来表征。

#### IDM：从前后变化中反推 latent action

IDM（Inverse Dynamics Model）同时看到 `F1` 与真实 `FT`，输出连续 posterior：

`z ~ q(z | F1, FT)`

`z` 不是关节角、末端位姿或夹爪开合，而是“当前视觉特征需要发生什么变化才能成为未来特征”的压缩代码。

IDM 使用 24 层、V-JEPA2 风格的时空 Transformer。它在训练时已经看见结果，所以其任务不是在未知未来中规划，而是编码已经发生的视觉转移。

#### LaWM：把抽象变化落到当前场景

LaWM 接收当前特征 `F1` 与 latent action `z`，输出预测未来特征 `FT_tilde`。它也是 24 层 Transformer，约 230M 参数，通过 AdaLN 注入 latent action。

AdaLN 相比简单加法注入更稳定：跨具身训练时 `z` 的范数变化较大，直接加到视觉 token 可能造成全局漂移和 loss spike。

#### Stage 1 损失

`L_stage1 = L_world + L_aux + β L_KL`，其中 `β = 1e−5`。

- `L_world = MSE(FT_tilde, FT)`：训练 IDM 产生对 decoder 有用的 `z`，训练 LaWM 预测未来特征。
- `L_aux = MSE(g(s, z), sT)`：用当前机器人状态和 `z` 预测 horizon 末端状态，迫使 `z` 包含具身运动，而不只是颜色/光照变化。辅助 head 在 Stage 1 后丢弃。
- `L_KL = KL(q(z|F1,FT) || N(0,I))`：限制 latent 容量、平滑 latent 空间，避免每个样本形成不可预测的离散代码，也让 Stage 2 的 policy prior 更容易拟合。

Stage 1 不需要真实动作标签，只需要前后视觉帧；因此可以用无动作标签的人类第一人称视频学习视觉动力学先验。

#### Stage 1 数据与成本

- 约 3,000 小时机器人视频；
- 约 1,500 小时第一人称人类视频；
- 合计约 4,500 小时；
- 16×H100，100K steps，global batch 1024；
- AdamW，学习率 `3e−4`，weight decay `1e−2`；
- 机器人视频 horizon 1.2 s，人类视频 horizon 0.4 s。

“efficient”主要指推理效率，不是轻量训练。

### 5.5 Stage 2：让 VLM 在看不到未来时驱动 LaWM

部署时真实未来 `FT` 不存在，IDM 无法使用。因此 Stage 2 训练 policy prior：

`z_hat = PolicyPrior(current images, instruction)`

模型使用 Qwen3-VL-2B 前 16 层、hidden size 1024，并加入 LA Query 和 Action Query。LA Query 聚合当前图像、指令与多视角上下文，经过轻量 projection 产生与 IDM latent 形状一致的 `z_hat`。

#### IDM teacher 与 VLM student

训练样本仍有真实未来，IDM 能产生教师 `z_teacher = IDM(F1, FT)`；VLM 只看当前与指令预测 `z_hat`。

两层约束解决“训练时 IDM、推理时 VLM”的接口替换：

1. **数值对齐**：`L_distill = MSE(z_hat, z_teacher)`；
2. **功能对齐**：`FT_hat = LaWM(F1, z_hat)`，再用 `L_subgoal = MSE(FT_hat, FT)`。

所以 VLM 输出不仅要“数值像 IDM”，还要能真正驱动同一个 LaWM 产生正确未来特征。

#### Action Expert

Action Expert 是 4 个 Alternate-DiT blocks、合计 16 层 Transformer。它读取两类信息：

- 语义流：VLM hidden states，回答“任务是什么、操作哪个对象”；
- 动力学流：当前 `F1` 与未来子目标 `FT_hat`，回答“当前状态怎样、下一阶段应变成怎样”。

Alternate-DiT 交替执行 Self Attention、Inverse-Dynamics Attention、Self Attention、Semantic Attention。动作从高斯噪声出发，经过默认 10 个 Flow Matching 去噪步，输出 EEF action chunk。

#### Stage 2 损失

`L_stage2 = 0.1 L_distill + 0.1 L_subgoal + L_action`

- `L_distill`：对齐 VLM 与 IDM 的 latent 坐标；
- `L_subgoal`：保证预测 latent 能经过 LaWM 产生正确未来；
- `L_action`：条件 Flow Matching，训练 Action Expert 从噪声生成示范动作块。

Stage 2 的 policy integration 只使用带语言指令的机器人轨迹；1,500 小时人类视频只在 Stage 1 影响 dynamics prior，不直接进入 Stage 2。该阶段训练 200K steps、global batch 1024、使用 64×H100；Action Expert 学习率 `1e−4`，其余可训练模块 `3e−5`。因此 “efficient” 指推理，不表示整体训练轻量。

### 5.6 Knowledge Insulation 与“Stage 2 是否冻结 LaWM”

论文明确的是：**Action Expert 的动作损失不能反向破坏 LaWM 的预训练动力学。** Knowledge Insulation 在动力学特征接口处阻断/隔离 `L_action` 的梯度。

但论文没有明确声明 Stage 2 对所有损失完全冻结 LaWM：

```text
L_distill → 更新 Policy Prior / VLM
L_subgoal → 更新 Policy Prior，并可能小学习率微调 LaWM
L_action   → 更新 Action Expert；被 KI 阻断，不破坏 LaWM
```

附录写 Action Expert 学习率 `1e−4`、其他模块 `3e−5`，更倾向于“`L_subgoal` 可能继续微调 LaWM”，但没有 optimizer parameter groups 或 `requires_grad` 配置，不能断言。

汇报时最严谨的说法：

- 确定：`L_action` 不更新 LaWM；
- 很可能：`L_subgoal` 可小学习率更新 LaWM；
- 不确定：作者实现是否把 LaWM 对所有 loss 完全冻结。

### 5.7 推理流程与效率来源

部署时不使用真实未来、IDM teacher、蒸馏或监督：

```text
1. F1 = DINO(current image)
2. z_hat = VLM(current images, instruction)
3. FT_hat = LaWM(F1, z_hat)       ← 单次前向
4. action chunk = ActionExpert(
       VLM context, F1, FT_hat, noise
   )                              ← 约 10 个动作去噪步
5. 执行动作块，获得新观测，再重复
```

LaWM future prediction 是单次前向，不迭代生成视频；Action Expert 仍然迭代去噪。论文报告的 187 ms 是完整 action-chunk 预测，不是只计算 LaWM。

### 5.8 混合控制频率与物理时间编码

不同数据集可能是 5/10/20 Hz。同一个 token index 对应的真实时间不同。LaWAM 固定动作块覆盖的物理时长 `τ`：

`H_b = round(τ × h_b)`

再给第 `i` 个动作 token 编码真实时间：

`t_i = i / h_b`

并加入 sinusoidal physical-time encoding。这样不同频率下相同真实时刻具有一致时间含义。附录用同一批 20 Hz LIBERO 轨迹下采样到 10/5 Hz，较干净地验证：混频无时间编码明显退化，加入编码后大部分恢复。

### 5.9 实验结果

#### LIBERO

该结果来自移除失败 demonstration 后的 benchmark 后训练：25K steps、global batch 256；40 个任务各评测 50 次，共 2,000 trials。LaWAM 在四个 suite 平均 98.6%，延迟 187 ms/action chunk。对比：

| 方法 | 类型 | 模型大小 | 延迟 | 平均成功率 |
|---|---|---:|---:|---:|
| π0.5 | VLA | 3.5B | 220 ms | 96.9 |
| Fast-WAM | Pixel WAM | 6B | 486 ms | 97.6 |
| Cosmos-Policy | Pixel WAM | 2.1B | 1413 ms | 98.5 |
| LingBot-VA | Pixel WAM | 5.5B | 4482 ms | 98.5 |
| LaWAM | Latent WAM | 2.3B | 187 ms | 98.6 |

正面结论是低延迟下保持高成功率；但 98.6 对 98.5 只有 0.1 个百分点。总计 2,000 次 trial 时约是 2 次成功的数量级，且没有多 seed/置信区间，不应强调“精度显著碾压”。

延迟在 A100、10 个动作去噪步下重复 1,000 次取均值，不含环境执行。187 ms 相比 LingBot-VA 的 4,482 ms 约快 24×；但不同 baseline 的软件栈和参数计数口径不完全统一。

#### RoboTwin 2.0

- 50 个双臂任务；
- 2,500 条 clean 与 25,000 条 heavy-randomization demonstrations；
- 后训练 100K steps，global batch 1024，64×H100，约 20 小时；
- 每任务在 clean/randomized setting 各评测 100 次；
- Clean 92.64，Randomized 89.80，两者平均约 91.22；
- Clean 第一，Randomized 不是第一；
- 与 Fast-WAM、LingBot-VA 平均值非常接近。

附录逐任务表暴露明显弱项：Hanging Mug 51/43、Open Microwave 41/43、Turn Switch 47/56，以及随机场景中的 Place Can Basket 65。说明 latent subgoal 没有消除精细接触、铰接约束和多阶段堆叠难题。

#### 真实机器人

每个任务 30 次：

| 方法 | Pick-and-Place | Open Drawer | Fold Towel | 平均 |
|---|---:|---:|---:|---:|
| π0.5 | 86.7 | 80.0 | 83.3 | 83.3 |
| Fast-WAM | 56.7 | 63.3 | 70.0 | 63.3 |
| LingBot-VA | 76.7 | 83.3 | 0.0 | 53.3 |
| LaWAM | 93.3 | 86.7 | 90.0 | 90.0 |

平台包括 Franka Panda 单臂与 Quanta X1 双臂，三项任务分别用 150、150、280 条真实 demonstrations 后训练。毛巾折叠约需 70 秒，低延迟可减少动态布料在推理等待期间继续运动造成的 stale observation。

但每任务仅 30 次，最小分辨率 3.33 个百分点，没有多次独立训练或置信区间。

#### 组件消融

Figure 6 比较 `w/o WM`、`w/o distill`、`w/o KI & distill`、`w/o pretrain` 与完整模型。移除 LaWM 的降幅最大，尤其影响 LIBERO-Long；移除 latent-action distillation 也明显下降，进一步支持 LaWM、蒸馏与 Knowledge Insulation 的互补作用。但正文只给图而无完整数值表，`w/o WM` 也未说明是否参数量匹配。

### 5.10 附录中最有意义的点

1. **架构与训练细节**：DINOv3 ViT-B/16；IDM/LaWM 各 24 层；LaWM 230M；Qwen3-VL 前 16 层；Action Expert 16 层；输入 256×256。
2. **不输入 proprioception**：可减少跨机器人状态定义不一致和 benchmark 捷径，但会丢失接触、力矩、关节极限与遮挡状态。
3. **物理时间编码受控实验**：同源轨迹下采样，是论文中因果隔离最干净的实验之一。
4. **RoboTwin 50 任务明细**：平均数之外能看到精细接触与随机化弱项。
5. **500 条 LIBERO rollout cosine 分析**：预测未来与真实未来保持较高相似，同时逐渐远离初始特征，说明 LaWM 不只是复制当前状态。
6. **Figures 11–13 的 subgoal 热图/PCA**：展示 action chunk 执行时机械臂逐渐靠近预测区域，但热图主要说明空间对应，不等于接触力学正确。
7. **Figures 14–15 跨具身 rollout**：同一 latent action 在不同初始场景产生上下文相关变化；这是表征层定性证据，不等于未见机器人零样本闭环成功。
8. **真实机器人协议**：所有方法共享固定测试初始配置，减少物理条件差异；失败分析指出高延迟与细微布料 feature resolution 的影响。

### 5.11 讨论重点

#### IDM 为什么看起来很强

因为 IDM 在训练时已经看见 `F1 + 真实 FT`。它不是预测未知未来，而是把已经发生的变化压缩成 `z`。真正困难的是 VLM 只看当前与指令，却要预测 IDM 看过未来后得到的 `z_teacher`。

#### IDM latent 与 VLM latent 能“一样”吗

- 张量形状一样：网络接口保证；
- latent 坐标系对齐：`L_distill` 强制；
- 具体数值不可能总一样：IDM 有未来信息，VLM 没有。

真正需要的是 `LaWM(F1,z_hat) ≈ FT`，不是逐位完全相同。

#### Posterior-prior gap

同一个当前状态和指令可能有多条成功未来。IDM 根据数据中实际发生的那一条给出 `z_teacher`；VLM 不知道示范选择了哪条路径。L2 蒸馏可能把多个模式平均。论文没有多 latent 采样、best-of-N、不确定性打分或 MPC，因此多模态未来仍是核心缺口。

#### latent action 会不会“作弊”

IDM 看见真实未来，可能把大量 `FT` 信息直接塞入 `z`，让 `z` 变成高带宽 future code，而非抽象动作。论文使用 bottleneck、KL、状态辅助损失和图像增强缓解，但没有清楚报告 `z` 的 token 数/维度，也没有 capacity sweep 或 mutual information 分析。

此外，`β=1e−5` 的 KL 只是很轻的正则；论文没有单独消融人类视频的贡献，也没有系统测试移动相机/强 ego-motion 下 latent action 是否把相机运动误当成环境动力学。

#### 它是不是经典世界模型规划

不是。推理时不枚举真实候选动作、不比较多条未来、不用 value 选择 rollout，只预测一个 `z_hat` 和一个 chunk-level subgoal。更准确的定位是：

> 显式未来 latent 特征条件化的 VLA，而不是成熟的反事实规划/MPC 世界模型。

### 5.12 汇报收束

LaWAM 最可靠的贡献是：**将 WAM 的“未来条件”从昂贵 RGB 视频压缩为一次前向得到的空间化 latent subgoal，在保持高控制成功率的同时显著降低推理延迟。**

最不应过度宣传的是：0.1 个百分点的 LIBERO 领先、跨具身定性图，以及“已经学到通用物理世界模型”。它更像高效的 future-feature-conditioned policy，而非通用 model-based planner。

<!-- CHUNK -->

## 六、VLA-OPD：学生自己走，Teacher 在学生走到的状态上纠错

### 6.1 基本信息与一句话结论

- 全名：**VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation**
- 首次公开：2026-03-27，arXiv:2603.26666v1，cs.RO，16 页。
- 第一/唯一作者单位：香港科技大学（广州）。
- Student：OpenVLA-OFT；Teacher：冻结的 SimpleVLA-RL。

<callout emoji="🎯" background-color="light-blue">一句话：让当前 Student 自己在环境中执行并暴露错误，再让冻结 Teacher 对 Student 实际访问的每个状态输出动作概率，最后用 Reverse-KL 把 Teacher 的软分布监督蒸馏给 Student。</callout>

### 6.2 原论文框架图

<image url="https://cdn.jsdelivr.net/gh/IRPN-LAB/VLA-OPD@main/figures/opd/figure1-1.png" width="1000" align="center" caption="VLA-OPD 原论文 Figure 1：Student on-policy sampling、Teacher dense labeling、Reverse-KL student optimization 三阶段循环。"/>

### 6.3 它如何桥接 SFT 与 RL

| 范式 | 训练状态来源 | 监督信号 | 主要问题 |
|---|---|---|---|
| Offline SFT | Expert | 专家动作，稠密 | 看不到 Student 自己走偏后的状态 |
| Online RL | 当前 Student | 环境成功/失败，稀疏 | 样本效率低、信用分配难 |
| VLA-OPD | 当前 Student | Teacher 动作分布，稠密 | 依赖强 Teacher 及其 OOD 可靠性 |

VLA-OPD 的关键不是把 SFT loss 换个名字，而是拆开两个问题：

```text
谁产生训练状态？当前 Student
谁提供监督？冻结 Teacher
```

这就是 On-policy / Interactive Policy Distillation。

### 6.4 “Student 初始化”具体是什么

Student 不是随机参数，也不是复制 Teacher：

1. 从预训练 OpenVLA-OFT checkpoint 开始，保留通用视觉、语言和动作先验；
2. 用目标任务少量示范做 SFT，得到表格中的 Student Init.。

数据条件：

- LIBERO：每个任务 1 条专家轨迹，不是所有任务总共 1 条；
- RoboTwin2.0：每个任务 1,000 条专家轨迹。

随机策略会长期停在无意义状态，Teacher 即使能标注也很难把它拉回任务区域；真实机器人还会有安全风险，所以需要基本能力初始化。

### 6.5 Teacher 从哪里来，隐藏成本是什么

论文 Teacher 是预先训练好的 SimpleVLA-RL，参考性能：

- LIBERO 平均 93.9%；
- RoboTwin2.0 平均 74.0%。

Teacher 在 OPD 过程中完全冻结。它读取 Student 访问的同一状态并输出 logits/log probability，但其动作不进入环境。

“1-demo Student 初始化”不包含 Teacher 的训练成本。Teacher 背后可能已有大规模专家数据、环境 rollout、RL 训练和 GPU 消耗。因此论文证明的是：

> 如何高效地把一个已存在强 Teacher 的闭环能力转移给 Student。

而不是“只凭 1 条示范从零获得 93%”。

### 6.6 三阶段训练循环

#### Phase 1：Student On-policy Sampling

当前 Student 采样动作并真正推动环境，得到包含正常、错误和恢复状态的轨迹。若改为 Teacher 执行，状态会重新接近专家分布，无法看到 Student 抓偏、推错、恢复中的状态。

#### Phase 2：Teacher Dense Labeling

对 Student 轨迹中的每个状态 `s_t`：

- Student 实际执行动作 `a_t`；
- 保存 Student 对该动作的 `log π_student(a_t|s_t)`；
- 冻结 Teacher 读取同一 `s_t`；
- 计算 Teacher 对 Student 已执行动作的 `log π_teacher(a_t|s_t)`。

Teacher 每一步都提供信号，避免仅用 episode 末端 0/1 奖励反推“哪一步错了”。

#### Phase 3：Reverse-KL Optimization

Student 目标是在自己访问的状态上最小化：

`D_KL(π_student || π_teacher)`

总体目标可写成：

`J(θ) = E_(s~d^π_student)[−D_KL(π_student(·|s) || π_teacher(·|s))]`

对 Student 采样动作定义稠密信号：

`r_OPD = log π_teacher(a|s) − log π_student(a|s)`

- Teacher 概率高于 Student：正信号，提高该动作概率；
- Teacher 概率低于 Student：负信号，降低该动作概率；
- 两者相等：信号为 0。

训练时对该 log-ratio stop-gradient，再乘 `∇ log π_student(a|s)` 更新 Student。Teacher 和环境都不需要反向传播。

对 Student 动作取期望时，`E[r_OPD] = −D_KL(Student||Teacher)`；单个采样动作的 reward 可以为正或负，但整体目标就是最小化 Reverse-KL。

### 6.7 Reverse-KL、Forward-KL 与 Hard-CE

论文定义：

- Forward-KL：`D_KL(Teacher || Student)`；
- Reverse-KL：`D_KL(Student || Teacher)`；
- Hard-CE：只回归 Teacher 的 top-1 动作。

#### Forward-KL

期望由 Teacher 分布计算，会要求 Student 覆盖 Teacher 有概率的全部区域。Teacher 在 Student 的 OOD 状态上若很不确定、分布很宽，Student 熵可能快速增大，论文观察到 entropy explosion 和明显 performance valley。

#### Hard-CE

把 Teacher 的软分布压成 argmax one-hot。相邻状态中 Teacher top-1 可能跳变，Student 熵快速下降，导致 premature entropy collapse。

#### Reverse-KL

优先检查 Student 当前会执行的动作是否被 Teacher 认可，强烈惩罚 Student 在 Teacher 低概率区域的概率质量。论文观察其成功率和熵更稳定。

但不要把 Reverse-KL 讲成“Teacher top-1 概率会一直增大”。信号看的是 Teacher/Student 概率比。若 Student 已把某动作提高到超过 Teacher，该动作会得到负信号；理论全局稳定点是 `Student = Teacher`，不是 one-hot Teacher argmax。

同样，Reverse-KL 不能凭空过滤掉 Teacher 不知道的错误知识。所谓 mode-seeking 更多是在多峰、有限表达和有限采样下的经验行为，不是一般性熵界。

从熵分解看，`−KL(Student||Teacher) = E_Student[log π_teacher] + H(Student)`：既鼓励 Teacher 认可的动作，也保留 Student 熵。这解释了它为何不像 Hard-CE 那样天然追逐单一 argmax，但不构成对所有设置的防坍缩保证。

### 6.8 Pure Distill 不是标准 GRPO

主实验配置为 batch size 64、每个 prompt group size `G=8`。多采样用于更好近似 Student 轨迹期望、平均环境随机性并降低 Monte Carlo 方差。

纯 Distill 阶段：

- 不使用环境成功奖励；
- 不按组内任务结果排名；
- 不训练 critic；
- 不计算 GRPO outcome-relative advantage；
- 直接平均 token/action-step 的 Reverse-KL 梯度。

这里的 `G=8` 只是 Monte Carlo 多采样/降方差，不做 outcome ranking；纯 Distill 也不需要 Critic。

论文的 **Distill + GRPO** 是先完成 OPD 蒸馏，再用真实环境成功/失败奖励做 GRPO。

为什么这样排：

1. 初始 Student 成功率低，直接 GRPO 常得到一组全失败轨迹；
2. 全失败时相对 advantage 接近 0，几乎无梯度；
3. OPD 先提供逐步 Teacher 信号，把 Student 拉到高成功区域；
4. 再用真实任务奖励继续优化，可减少纯蒸馏受 Teacher 上限限制。

### 6.9 从严格算法角度，它属于什么

它使用 Student on-policy rollout、采样动作和 score-function `∇logπ` 更新，形式上与策略梯度相似；但主要监督来自 Teacher 分布，不是环境 return。

论文公式对每个时刻只使用当前 `r_t^OPD`，没有把未来 KL 奖励做 reward-to-go 传回当前动作。因此更准确的训练理解是：

```text
用当前 Student 采集一批 on-policy 状态
→ 暂时把状态视为固定数据
→ 在每个状态上最小化条件动作分布 Reverse-KL
→ 更新 Student
→ 重新采样新状态
```

它更接近**交替式在线蒸馏 / 软标签 Reverse-KL DAgger**，而不是对完整 MDP 状态占用目标求精确 RL 梯度。

复现上还有一个关键歧义：论文称 token-level dense supervision，但一个环境动作可能包含多个 action token；v1 未说明 reward 按 token、环境步还是 action chunk 聚合，也未给 tokenizer 与采样温度。

### 6.10 实验结果

#### LIBERO

| 方法 | Spatial | Object | Goal | Long | 平均 |
|---|---:|---:|---:|---:|---:|
| Student Init. | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| VLA-OPD Distill | 84.3 | 93.8 | 92.5 | 78.9 | 87.4 |
| Distill + GRPO | 93.4 | 95.3 | 94.5 | 90.2 | 93.4 |
| Teacher | 94.2 | 96.1 | 94.6 | 90.7 | 93.9 |

同一 Student 管线内最有意义的变化是 `48.9 → 87.4 → 93.4`。

- 纯 Distill 提升 38.5 个百分点，关闭约 85.6% 的 Teacher 差距；
- Distill + GRPO 关闭约 98.9%；
- LIBERO-Long 从 17.3 到 78.9，提升 61.6 个百分点，是最符合“纠正复合误差”动机的结果。

“+38.5%”更准确应写为“+38.5 percentage points”，不是相对提升 38.5%。

#### RoboTwin2.0

| 方法 | Pick Dual Bottles | Place Empty Cup | Handover Block | Stack Bowls Two | 平均 |
|---|---:|---:|---:|---:|---:|
| Student Init. | 29.7 | 77.3 | 33.1 | 40.6 | 45.2 |
| VLA-OPD Distill | 66.4 | 90.6 | 52.3 | 75.0 | 71.1 |
| Teacher | 68.3 | 94.2 | 57.8 | 75.8 | 74.0 |

纯 Distill 提升 25.9 个百分点，关闭约 89.9% Teacher 差距。但 Handover Block 仍差 5.5 个百分点，复杂双臂长时程协调未完全蒸馏。

值得强调的是，RoboTwin Student 已使用每任务 1,000 条离线 demonstration，初始化平均仍只有 45.2%；这说明单纯增加离线示范并没有消除闭环状态分布偏移。

#### 训练效率

- LIBERO-Object：约 10 个 optimizer step 超过 90%；
- LIBERO-Long：约 50 step 接近 80%，GRPO 约 150 step；
- 作者称约 3× speedup。

但一个 OPD step 包含多条环境 rollout、每状态 Teacher 推理和 Student/Teacher logits。论文没给 environment frames、Teacher query、GPU hours、wall-clock 与 FLOPs，所以 3× 只是更新步数加速，不等于总算力或墙钟节省 3×。

#### 遗忘与消融

- 4 个 held-out unseen tasks（2 个 Object、2 个 Spatial）上，Offline SFT 随 seen performance 提高而明显遗忘，部分 Object unseen 甚至接近 0；VLA-OPD/在线 RL 保留更好；
- 但只有 4 个任务、无数值表、无多 seed/置信区间，不能推广为一般 continual learning 保证；
- Reverse-KL/Forward-KL/Hard-CE 动态只展示 RoboTwin 的 Beat Block Hammer；Forward-KL 早期有超过 50% performance valley；
- `G={2,4,8}` 消融使用 batch 32，而主实验为 64；G=8 最平滑、最终约 89%，但未在相同总 rollout 预算下比较。

### 6.11 附录与复现状态

当前 v1 没有 Appendix 或独立 supplementary：第 1–13 页为正文，后面是参考文献。正文明确给出 Student/Teacher、初始化示范数、batch 64、G=8 等，但缺少：

- optimizer、学习率、scheduler、weight decay、gradient clipping；
- 学生具体冻结/更新哪些层；
- action tokenization、采样温度、rollout horizon；
- checkpoint 评测回合数、种子、标准差；
- Teacher query 数、环境步数、GPU/墙钟；
- Teacher logits 的 clipping 与数值稳定处理；
- mixed precision、episode 截断方式与 Teacher checkpoint/训练预算；
- 真实机器人安全机制。

截至 2026-07-15，项目页仍标注 Code Coming Soon。因此目前可以理解算法并实现近似版本，但不能确认作者的精确工程细节。

### 6.12 讨论重点

1. **On-policy 的准确含义**：Student 的动作真正推动环境；Teacher 只标注，不接管。Teacher 提供标签不改变轨迹属于 Student on-policy 的事实。“On-policy”描述数据来自当前 Student；“online”描述数据通过当前环境交互获得，两者相关但不是同义词。
2. **Teacher OOD 可靠性**：方法专门收集 Student 失败状态，但这些状态也可能超出 Teacher 分布。若 Teacher 高熵、校准差或直接给错，蒸馏会复制问题。
3. **Reverse-KL 不是魔法不确定性过滤器**：全局最优仍是 Student 匹配 Teacher；Teacher 完全均匀时，Student 也会匹配均匀。
4. **低概率模式难恢复**：动作从 Student 分布采样，某个有效动作若已被压到极低概率，可能长期采不到，也就得不到恢复梯度。
5. **“1-demo”不能脱离上下文**：完整条件是少量 Student SFT + 强 RL Teacher + 大量 Student 在线 rollout。
6. **真实机器人缺失**：Student 失败探索在真实系统中需要碰撞约束、人类接管、复位成本和 Teacher 延迟管理。
7. **Teacher 接口要求**：Reverse-KL 需要 Teacher 提供 logits/log-probability；只返回 top-1 动作的黑盒 API 不能直接实现本文算法。

### 6.13 汇报收束

VLA-OPD 最有价值的洞见是：

> “On-policy”不必和“稀疏环境奖励”绑定；可以让 Student 产生真实部署状态，同时让 Teacher、规则或其他模型提供稠密监督。

它是思路清晰、增益显著、工程价值很高的交互式 VLA 后训练框架。但它转移的是已有强 Teacher 的能力，没有消除 Teacher 成本；Reverse-KL 的理论表述和真实机器人可扩展性仍需要更多证据。

<!-- CHUNK -->

## 七、四篇论文放在一起：最容易讲清楚的比较

### 7.1 信号、环境和更新对象

| 论文 | 训练状态/轨迹来自哪里 | 主要学习信号 | 更新谁 | 训练时是否需真实在线环境 | 部署时是否保留 World Model |
|---|---|---|---|---|---|
| VLA-RFT | 离线起点 + 学习到的 World Model rollout | 与专家参考视觉后果的 L1+LPIPS | Flow action head + Sigma Net | 否，RFT 主要在 World Model 中 | 否 |
| WAM-RL | Actor 在 LIBERO/RLBench 在线执行 | 预测视频 vs 执行视频一致性；成功视频做 SFT | Actor + World Model（两种 loss） | 是 | 是 |
| LaWAM | 离线机器人/人类视频与带语言机器人轨迹 | 未来 DINO 特征、latent 蒸馏、动作 Flow Matching | Stage 1：IDM+LaWM；Stage 2：VLM+Action Expert，LaWM 可能小学习率微调，IDM 作 Teacher | 否 | 是，保留 LaWM |
| VLA-OPD | Student 在仿真环境的 on-policy rollout | Teacher/Student log-probability ratio | Student | 是 | 无额外 World Model |

### 7.2 四篇论文里的“未来”不是一回事

- **VLA-RFT**：未来是候选动作在 World Model 中产生的预测 RGB 轨迹，用来算训练奖励。
- **WAM-RL**：未来是 World Model 的视觉计划；Actor 要把它在环境中实现，同时成功执行视频反过来更新 World Model。
- **LaWAM**：未来是动作块终点的 DINO feature map，不是可观看 RGB 视频，直接作为动作条件。
- **VLA-OPD**：不显式建未来模型；通过 Student 自己走到的状态，让 Teacher 给当下动作纠错。

### 7.3 哪些属于 RL

- **VLA-RFT**：优化形式上是真正 policy-gradient/GRPO，但任务目标仍是专家轨迹匹配。
- **WAM-RL**：Actor 是 policy-gradient RL；World Model 的 Online Video SFT 不是 RL。
- **LaWAM**：核心训练是监督/自监督 latent world modeling、蒸馏和 Flow Matching，不是 RL。
- **VLA-OPD**：使用 on-policy sampling 和 `∇logπ`，但主要信号来自 Teacher，更准确叫 on-policy policy distillation / interactive imitation learning；可选后续 GRPO 才直接使用环境任务奖励。

### 7.4 哪篇的探索空间最大

不能只看有没有噪声：

1. VLA-RFT 有 SDE 和多候选，但奖励逐帧贴专家轨迹，属于专家附近局部搜索。
2. WAM-RL 有真实环境交互和 Flow-SDE，但重建奖励要求符合 World Model 想象，探索受视觉计划约束。
3. LaWAM 默认只预测一个 latent subgoal，没有 best-of-N/MPC，属于单计划闭环执行。
4. VLA-OPD 的 Student 会真实访问自身错误状态，但优化目标是靠近 Teacher，能力上限与 OOD 正确性仍由 Teacher 决定；追加 GRPO 后才有机会按任务结果超越 Teacher。

### 7.5 实验证据强弱

| 论文 | 最强证据 | 最大证据缺口 |
|---|---|---|
| VLA-RFT | LIBERO 标准与扰动均提升，reward 消融清楚 | 无真实机器人、无多 seed、World Model OOD/reward correlation 未测 |
| WAM-RL | Actor-only 与联合更新方向对比 | 长时程仅一个低成功率任务，组件消融不干净，复现超参缺失 |
| LaWAM | 仿真 + 三项真实机器人 + 明显延迟优势 | 精度差距小、跨具身主要定性、latent 容量与多模态未来未解决 |
| VLA-OPD | 同一 Student 管线大幅提升，长时程收益突出 | Teacher 成本与 OOD 可靠性、无真实机器人、无附录/代码/完整计算口径 |

## 八、建议的现场讲述顺序

### 8.1 开场：先讲共同矛盾

“普通 VLA 的 SFT 只在专家状态上训练。真正部署时，状态由模型自己的历史动作产生，所以四篇都在想办法让训练接触‘模型会走到哪里’或显式建模‘动作会带来什么未来’。”

### 8.2 每篇都用同一个四问模板

1. 谁产生训练状态？
2. 谁评价动作/未来？
3. 哪个模块接收梯度？
4. 评价信号是否就是任务成功？

这样可以避免陷入名词堆叠。

### 8.3 每篇的一句话记忆钩子

- **VLA-RFT**：在冻结 World Model 里，围绕专家后果做 GRPO 局部搜索。
- **WAM-RL**：RL 训练 Actor 实现想象，成功视频训练 World Model 改进想象。
- **LaWAM**：用一个未来 DINO latent 子目标替代昂贵视频生成。
- **VLA-OPD**：Student 自己走，Teacher 在 Student 状态上逐步纠错。

### 8.4 最后用三条批判性结论收束

1. **World Model reward 不自动等于真实任务 reward**：VLA-RFT 可能复制专家轨迹，WAM-RL 可能“自洽但失败”。
2. **On-policy 不自动等于开放式探索**：VLA-OPD 仍受 Teacher 限制，VLA-RFT/WAM-RL 仍受参考轨迹或视觉计划限制。
3. **Efficiency 要看完整口径**：400 updates、3× steps、187 ms inference 分别是不同口径，不能互相类比，更不能忽略预训练、Teacher 或视频模型成本。

<callout emoji="✅" background-color="light-green">最终主线：四篇论文分别把“未来后果、在线执行、latent 子目标、Teacher 纠错”接入 VLA 后训练。真正的差别不在于都用了哪些热门名词，而在于监督信号是否与任务成功一致，以及这种信号能否覆盖 Student 自己造成的闭环状态。</callout>

## 九、原论文与项目链接

- VLA-RFT：[arXiv](https://arxiv.org/abs/2510.00406)；[项目页](https://vla-rft.github.io/)
- WAM-RL：[arXiv](https://arxiv.org/abs/2606.17906)
- LaWAM：[arXiv](https://arxiv.org/abs/2606.15768)；[项目页](https://rlinf.github.io/LaWAM/)
- VLA-OPD：[arXiv](https://arxiv.org/abs/2603.26666)；[项目页](https://irpn-lab.github.io/VLA-OPD/)
