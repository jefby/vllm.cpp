# 调试记录：sm_110 / QNX 上 Qwen3.6-35B-A3B Q4_K_M 挂起

> 状态：**待分析**（2026-08-24 记录，次日继续）
> 环境：NVIDIA Jetson Thor 级 sm_110（compute capability 11.0），**QNX 7.2.4**
> 现象：`qwen3.8-27B`、`qwen3.6-4B` 均正常；仅 `qwen3.6-35b-a3b` **Q4_K_M** 挂起（hang）

---

## 1. 已排除的范围（关键结论）

27B dense 与 4B 正常运行证明以下路径在 sm_110/QNX 上是好的：

| 已验证正常 | 说明 |
|---|---|
| CUDA 后端基础 | pinned 内存（`cudaHostAlloc`）、事件、stream/device 同步 |
| QNX 驱动栈 | cuBLASLt 初始化与 heuristic 查询（dense GEMM 走到了） |
| GGUF Q4_K quant-dot | `vt::MatmulBTQuant`（若 27B 也是 Q4_K_M 加载） |
| 标准 attention 内核 | portable FA-style prefill + paged decode |
| 引擎调度主循环 | add_request → schedule → forward → sample |

**hang 被隔离到 35B-A3B 独有的路径（2026-08-25 根据 upstream #1880 记录修正后）：**

1. **MoE 分组专家 GEMM**：`vt::MatmulBTQuantGrouped`（`src/vt/cuda/cuda_quant_dot.cu`，2211 行，无 `__CUDA_ARCH__` 守卫 = portable 内核）
2. **GDN hybrid 层 prefill 内核**：`src/vt/cuda/cuda_gdn.cu` 的 chunked-scan prefill（eager 执行）
3. **decode graph capture 期间的驱动交互**（#1732 同类）：MoE 有 decode graph
   （`qwen3_5_moe.cpp:176` 创建），capture 内的 cuBLASLt heuristic/驱动调用在 QNX 上可能表现为 hang 而非 throw

> **修正记录（#1880 分析结论）**：初版曾把「GDN packed decode 回退」列为嫌疑 #2。
> 已排除：upstream #1793 确认 **GGUF MoE loader 仍保持 `in_proj_b`/`in_proj_a`
> split**（`qwen3_5_gguf_weights.cpp:1082-1085`），`has_packed_ba` 为 false，
> packed GDN decode 在 GGUF MoE checkpoint 上**根本不会到达**——Triton cubin
> 回退路径不是本 hang 的源。

## 2. 为什么这两条路径可疑

- 项目对 sm_110 的 runtime 验证记录见 `.agents/backend-matrix.md` 行 `BACKEND-CUDA-SM110`：
  仅跑过 **Llama-3.2-1B bf16（dense attention，无 MoE 无 GDN）12/16 strict token-exact**，
  以及 marlin-nvfp4 数值门。MoE/GDN 路径在 sm_110 上是**首跑**。
- Triton AOT cubin 只有 `triton_aot_vendored/sm_121a/` 一个版本；sm_110 运行时按精确 SM
  选 tree 失败 → **静默回落到手写 CUDA 内核**（`cuda_gdn.cu:2714` `TryTritonPackedDecode`
  失败后走 hand kernels）。回落路径从未在真实 sm_110 上执行过。
- 兄弟架构已知先例：**sm_87（Orin）上 async runner 路径崩溃**（illegal memory access），
  只有同步路径验证通过——非 GB10 架构的 async/specialized 路径有出 bug 的历史。
- GDN 内部有串行回退实现（`cuda_gdn.cu:4721` 串行前向替换），串行长循环是最典型的挂起点。

## 3. 嫌疑点排序

| # | 嫌疑点 | 锚点 | 状态 |
|---|---|---|---|
| 1 | GDN prefill chunked-scan 的 portable 内核挂死/死循环 | `src/vt/cuda/cuda_gdn.cu`（含 `:4721` 串行前向替换） | 待验证 |
| 2 | ~~GDN packed decode 手写内核~~ | — | **已排除**（GGUF MoE 不走 packed leg，#1793） |
| 3 | MoE grouped quant-dot 内核 | `src/vt/cuda/cuda_quant_dot.cu` | 待验证 |
| 4 | 加载期 device_fit 决策异常（QNX 上 cudaMemGetInfo 上报差异 → keep-quant/offload 误判） | `src/vllm/model_executor/model_loader/gguf_device_fit.cpp` | 待验证 |
| 5 | **graph capture 期间的 cuBLASLt/驱动交互**（#1732 同类；GB10 上是 throw，QNX 可能是 hang） | `qwen3_5_moe.cpp:176`；upstream #1741 + `VT_FP8_PLAN_CACHE=1` 才双修复 | 待验证（新增） |
| 6 | 引擎级可复现性问题（#1877：GB10 NVFP4 35B 同臂重复运行即分歧行） | upstream issue #1877 | 背景风险 |

## 4. 明天的调试计划（按序执行）

```sh
# ① 定位 hang 阶段：加载中？第一个 token（prefill）？decode 几步后？
VT_ENGINE_STEP_LOG=1 ./server --model <35b-q4_k_m.gguf> ... 2>&1 | tee run.log
#   - 无任何 step 输出且权重已加载完 → 首次 forward 内核挂死（嫌疑 1/2/3）
#   - 加载进度都没走完 → 嫌疑 4

# ② 强制同步引擎路径（排除 async runner —— sm_87 先例）：
VT_ASYNC_RUNNER=0 ...

# ③ 二分 GDN 内核变体（每次只改一个开关，看 hang 是否消失）：
VT_ASYNC_RUNNER=0 VT_GDN_PACKED_DECODE=0 ...
VT_ASYNC_RUNNER=0 VT_GDN_FUSED_DECODE=0 ...
VT_ASYNC_RUNNER=0 VT_GDN_VALIDATE=1 ...        # 打开形状校验

# ④ 若加载即 hang，显式给设备拟合预算：
VT_DEVICE_WEIGHT_BUDGET_BYTES=<bytes> ...

# ⑤ graph capture 二分（#1880/#1732 分析新增的高优先级开关）：
VLLM_CPP_CUDAGRAPH=0 VT_ASYNC_RUNNER=0 ...
#    - hang 消失 → 问题在 capture 期间的 cuBLASLt/驱动调用（嫌疑 5，#1732 同类）

# ⑥ 对照实验：同一模型在 GB10 (sm_121a) 上跑同版本二进制确认无回归；
#    再试 35B-A3B 的 NVFP4 safetensors 版本（走 Marlin + 可达 packed leg）
#    以区分：NVFP4 不 hang 而 GGUF hang → 指向 quant-dot/device_fit（嫌疑 3/4）；
#            两者都 hang → 指向 GDN prefill 或 capture（嫌疑 1/5）。
```

### 有用的环境开关速查（grep 自 `getenv("VT_*")`）

- 引擎：`VT_ENGINE_STEP_LOG`、`VT_ASYNC_RUNNER`、`VT_DEBUG_SAMPLED`
- GDN：`VT_GDN_VALIDATE`、`VT_GDN_PACKED_DECODE`、`VT_GDN_PACKED_DECODE_TRITON`、
  `VT_GDN_FUSED_DECODE`、`VT_GDN_CHUNKED`、`VT_GDN_DIAG_STEP_LOG`
- 加载：`VT_DEVICE_WEIGHT_BUDGET_BYTES`、`VT_DIRECT_DEVICE_LOAD`

## 5. 支持面事实备查

- sm_110 构建：单架构 `-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=OFF`，
  nvcc 13，portable-kernels-only（fp4-mma/cutlass-nvfp4/cutlass-fp8/fa2 均 DISABLED，
  仅 `marlin-nvfp4` 自 2026-08-11 ENABLED）。
- 35B-A3B 注册名 `Qwen3_5MoeForConditionalGeneration`（`qwen3_5_moe.cpp:91`）；
  GGUF 入口 `LoadQwen3_5MoeFromGguf`（`qwen3_5_gguf_weights.cpp:1312`）。
- 本仓库此前对 Thor sm_110 的验证全部基于 JetPack R38（Linux + driver 580），
  **QNX 是全新宿主 OS，零验证零适配代码**。

## 6. 参考提交与 issue（2026-08-24 upstream 同步合入）

- `1724be38e` record(GDN-MOE-PACKED-BA) #1880：packed GDN decode 在 35B 上首次到达并测量
  （spec：`.agents/specs/gdn-moe-packed-ba.md`）；packed 快 ~0.8%（bf16）/ ~3.5%（NVFP4）；
  遗留 #1877 / #1878 / #1793 三个 owed。
- `732e9ddf8` fix(FIX-FP8-PLAN-CAPTURE-1843)：fp8 plan cache 默认 ON（CUDA 13.3 下
  capture 时未缓存 lane 有错）——重新构建时用最新 HEAD 保证行为一致。
- 关键推论链：GGUF MoE split loader（#1793）→ packed leg 不可达 → 本 hang 与 Triton
  cubin/packed-decode 无关；MoE decode graph 存在 → capture 期驱动交互成为新嫌疑。

## 7. 后续动作

- [ ] 按 §4 抓取 hang 阶段证据（step log + 二分开关结果）
- [ ] 结果回填本文档 §3 排序表（确认/排除）
- [ ] 开 GitHub issue 记录该发现（sm_110/QNX 首跑 MoE+GDN hang，untested-portable-fallback）
- [ ] 若定位到具体内核，评估是否需要 sm_110 专属守卫或修复

---
*记录人：ox-alpha（agent）；分析基于 dev@faf07978e（upstream main 3e4cd6d11 同步后）。*
