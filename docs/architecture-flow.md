# vllm.cpp 软件架构流程分析

> 生成时间：基于仓库当前 `main` 状态（2026-08-05 会话快照）。
> 分析范围：C++ 推理引擎核心、vt 张量运行时、调度与执行管线、服务入口与 C ABI。

---

## 1. 总体定位与设计哲学

`vllm.cpp` 是 vLLM V1 / Model Runner V2 的纯 C++ 移植目标：

- **无 Python / 无 PyTorch / 无 ggml 依赖**：推理时不带解释器，二进制仅 ~66 MiB。
- **行为镜像 vLLM**：同一 workload、同一模型、greedy 解码要求 token-for-token 一致。
- **llama.cpp 式部署**：核心产物是 `libvllm` + 稳定 C ABI（`include/vllm.h`），上层再包 CLI / OpenAI 服务器。
- **GGUF 一等公民**：与 safetensors 并列支持，CPU 可直接在量化块上计算。
- **多后端同构**：CPU、CUDA、Metal、Vulkan、XPU 等通过统一 `vt::` 运行时接入，引擎代码不感知具体硬件。

---

## 2. 分层架构全景

```text
┌─────────────────────────────────────────────────────────────────────┐
│  入口层 (Entrypoints)                                               │
│  examples/cli          → libvllm C ABI (vllm_complete / streaming)  │
│  examples/server       → OpenAI API (/v1/completions, /v1/chat/...) │
│  C/C++/Go/Rust FFI     → vllm_engine_load / vllm_chat / ...         │
├─────────────────────────────────────────────────────────────────────┤
│  服务与同步层 (Serving / Frontend)                                  │
│  vllm/v1/engine/async_llm.h  → 异步请求收集 + 输出处理线程          │
│  vllm/v1/engine/llm_engine.h → 同步离线 LLMEngine                   │
│  vllm/entrypoints/openai/*   → HTTP 路由、chat template、工具解析   │
├─────────────────────────────────────────────────────────────────────┤
│  引擎核心 (Engine Core)                                             │
│  vllm/v1/engine/core.h       → schedule / execute / sample / update │
│  vllm/v1/core/sched/scheduler.h → token-budget 连续批调度器         │
│  vllm/v1/core/block_pool.h, kv_cache_manager.h → Paged KV + 前缀缓存│
├─────────────────────────────────────────────────────────────────────┤
│  执行器与模型运行器 (Executor / Model Runner)                       │
│  vllm/v1/executor/executor.h → 透传 SchedulerOutput 到 Runner       │
│  vllm/v1/worker/gpu/runner.h → GPUModelRunner, InputBatch, Sampler  │
│  vllm/v1/attention/backends/* → GDN / Paged / MLA 注意力元数据      │
│  vllm/model_executor/models/* → 各模型 Forward 实现                 │
├─────────────────────────────────────────────────────────────────────┤
│  张量运行时 (vt:: Tensor Runtime)                                   │
│  include/vt/tensor.h, backend.h, device.h, ops.h                    │
│  src/vt/ops.cpp, op_provider.cpp → 显式输出、按设备分发的 op 表     │
│  src/vt/cuda/*, cpu/*, metal/*, vulkan/* → 各后端内核实现           │
├─────────────────────────────────────────────────────────────────────┤
│  加载与量化 (Loading / Quantization)                                │
│  model_loader: safetensors + GGUF 双路径                            │
│  NVFP4 W4A4、FP8、GGUF Q4_0/Q8_0/Q3_K/... 直接计算或反量化          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 入口与使用路径

### 3.1 C ABI（库入口）

`include/vllm.h` 是稳定的 C 接口，当前 ABI 版本 `VLLM_ABI_VERSION 10`：

- `vllm_engine_load()`：从模型目录或 **GGUF 文件**构建完整引擎（`model_path` 直接接受 `.gguf`）。
- `vllm_complete()` / `vllm_complete_stream()`：阻塞式或流式完成。
- `vllm_request_submit()` / `vllm_request_wait()`：非阻塞异步请求。
- `vllm_chat()` / `vllm_chat_stream()`：OpenAI 风格 chat，引擎侧做 chat template、tool/reasoning 解析。
- 结构化输出、投机解码、前缀缓存、调度策略、外部 KV 传输均通过 `vllm_model_params` 字段配置。

### 3.2 CLI

`examples/cli/main.cpp` 完全走 C ABI：

```text
解析参数 → vllm_model_params_default() → vllm_engine_load()
    → vllm_complete_stream() / vllm_complete() → 输出文本
```

### 3.3 OpenAI 服务器

`examples/server/main.cpp` 直接拼装 C++ 引擎栈，提供 **OpenAI 兼容 HTTP API**：

```text
命令行参数
    → LoadChatTemplate / HfConfig / Tokenizer
    → LoadedEngine::FromModelDir() 构造 {LLMEngine | AsyncLLM}
    → api_server + serving_chat / serving_completion
    → cpp-httplib 处理 HTTP 请求
```

已实现的端点：

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/v1/completions` | 文本补全 |
| POST | `/v1/chat/completions` | 聊天 + tools / reasoning / streaming |
| GET  | `/v1/models` | 列出模型 |
| GET  | `/health`、`/ping`、`/version` | 健康检查 |
| GET  | `/metrics` | Prometheus 指标（vLLM 同名） |
| POST | `/tokenize`、`/detokenize` | 分词/反分词 |
| POST | `/reset_prefix_cache` | 重置前缀缓存 |
| GET  | `/server_info` | 服务信息 |

启动示例：

```sh
build/examples/server --model /path/to/Qwen3.6-27B --port 8000 --max-num-seqs 32
```

任意 OpenAI 客户端直接连接即可：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
client.completions.create(model="Qwen3.6-27B", prompt="Hi", max_tokens=64)
```

---

## 4. 引擎核心执行流程

### 4.1 同步路径：`LLMEngine`

```text
add_request(prompt, sampling_params)
    → InputProcessor::process_inputs()      // 分词、 multimodal 预处理
    → OutputProcessor::add_request()        // 注册输出收集器
    → EngineCore::add_request(Request)      // 注入调度器

step() 循环
    → EngineCore::step()
        1. scheduler_.schedule()            // 产出 SchedulerOutput
        2. executor_.execute_model()        // 前向：更新 InputBatch → 模型 Forward → stash logits
        3. scheduler_.get_grammar_bitmask() // 结构化输出掩码
        4. executor_.sample_tokens()        // 采样 → ModelRunnerOutput
        5. scheduler_.update_from_output()  // 追加 token、检测 stop/EOS、释放 KV 块
    → OutputProcessor::process_outputs()    // 解码文本、finish_reason、tool_calls
    → 返回 RequestOutput
```

### 4.2 异步服务路径：`AsyncLLM`

```text
add_request()
    → 同 LLMEngine 前端处理
    → 通过 InprocClient 把 Request 推入 EngineCoreProc 的队列

EngineCoreProc 线程
    → step_with_batch_queue()              // batch-queue depth=2 时启用异步调度
    → 前向与采样输出 copy 重叠，降低 GPU 空闲

输出处理线程
    → 拉取 EngineCoreOutputs
    → OutputProcessor 生成每个请求的 delta
    → 唤醒对应 RequestOutputCollector
```

---

## 5. 调度器：统一 Token-Budget 连续批

`vllm/v1/core/sched/scheduler.h` 实现 vLLM V1 的核心调度思想：**没有显式的 prefill/decode 阶段**，每个请求只有 `num_computed_tokens` 和 `num_tokens`，每步在 token budget 内让 `num_computed` 追上 `num_tokens`。

### 5.1 schedule() 主要步骤

```text
1. new_step_starts()                    // KV cache manager 进入新步
2. 先调度 RUNNING 队列                 // 运行中请求优先
   - 计算 num_new_tokens = num_tokens_with_spec + placeholders - num_computed
   - 受 long_prefill_token_threshold、token_budget、max_model_len 限制
   - allocate_slots() 失败则 FCFS 尾部抢占 (preempt_request)
3. 再调度 WAITING 队列                 // 无抢占且仍有预算时
   - get_computed_blocks() 命中前缀缓存
   - 新请求按 FCFS / priority / lpm（SGLang 最长前缀匹配）顺序准入
4. 组装 SchedulerOutput
   - new_reqs: 新/恢复请求，携带 prefill_token_ids（MRV2 合同）
   - cached_reqs: 已运行请求的 num_computed + new_block_ids 差分
   - num_scheduled_tokens、spec_decode_tokens、common_prefix_blocks 等
5. _update_after_schedule()             // 推进 num_computed、设置 is_prefill_chunk
```

### 5.2 KV Cache 与前缀缓存

- `BlockPool`：管理固定大小的 KV 块，支持引用计数、分配、释放、前缀哈希链。
- `KVCacheManager`：按 KV cache group（如 full-attention group + GDN group）分配块。
- 前缀缓存：对 dense full-attention 模型默认开启；hybrid / GDN 模型默认关闭。
- 抢占：显存不足时把运行中请求换出（preempt），恢复时作为新请求重新准入。

---

## 6. 模型执行器与 Runner

### 6.1 Executor

`vllm/v1/executor/executor.h` 是薄层，把 `SchedulerOutput` 转给 `GPUModelRunner`：

```text
execute_model(scheduler_output)
    → runner_.execute_model(scheduler_output)   // 前向，返回 nullopt（MRV2 合同）
sample_tokens(grammar_output)
    → runner_.sample_tokens(grammar_output)     // 采样，返回 ModelRunnerOutput
```

### 6.2 GPUModelRunner

`vllm/v1/worker/gpu/runner.cpp` 是执行核心：

```text
execute_model(scheduler_output)
    1. input_batch_.update_states(scheduler_output)      // 接纳新请求 / 应用 cached diff
    2. reorder_batch_to_split_decodes_and_prefills()     // decode-first 重排
    3. prepare_inputs()                                  // token_ids / positions / attn_meta
    4. 根据模型类型调用 Forward：
       - Qwen3_5Model::Forward (MoE hybrid GDN + full-attn)
       - Qwen3_5DenseModel::Forward
       - Llama / Mistral / Gemma / DeepSeek-V4 / ... 等注册模型
    5. stash logits + step inputs

sample_tokens(grammar_output)
    1. 按 logits_indices 收集 [num_reqs, vocab] 行
    2. apply_grammar_bitmask()                           // 结构化输出 -inf 掩码
    3. Sampler::forward()                                // greedy / temperature / top-k / top-p / penalties
    4. 写回 input_batch_.output_token_ids / last_sampled_tokens
```

### 6.3 InputBatch

`vllm/v1/worker/gpu/input_batch.h` 是跨步持久化的 per-slot 状态容器：

- `req_ids`、`token_ids_cpu`、`num_computed_tokens_cpu`、`num_prompt_tokens`
- 每请求的 `block_ids`（多 KV group）
- 采样参数：`temperature`、`top_p`、`top_k`、各类 penalty
- 支持 `add_request / remove_request / condense / swap_states`

### 6.4 注意力后端

| 后端 | 路径 | 说明 |
|------|------|------|
| GDN | `vllm/v1/attention/backends/gdn_attn.h` | GatedDeltaNet：chunked-scan prefill + 递归 decode |
| Paged Attention | `vllm/v1/attention/backends/` | full-attention 层的 FA2-style varlen prefill + paged decode |
| MLA | MiniCPM3 / DeepSeek-V2 / V4 | MLA decode/prefill、缓存吸收 |

GDN 元数据负责把 batch 分割成 `num_decodes / num_prefills / num_spec_decodes`，并生成 `query_start_loc`、`state_indices`、`has_initial_state` 等，驱动 `vt::GdnPrefill / GdnDecode / GdnPackedDecode`。

---

## 7. vt:: 张量运行时

`vt::` 是 vllm.cpp 自有的显式输出 eager 张量运行时，替代 torch/Triton。

### 7.1 核心抽象

- `vt::Tensor`：`{data, dtype, device, rank, shape[4], stride[4]}` 的非 owning 视图，最大 4 维。
- `vt::Device`：`{DeviceType, index}`，支持 CPU / CUDA / METAL / VULKAN / XPU。
- `vt::Queue`：执行队列（CUDA stream / Metal command queue 等），带单调 ID。
- `vt::Backend`：设备后端接口，提供 `Alloc / Free / Copy / CreateQueue / Synchronize`、事件、pin 内存、graph capture 等。

### 7.2 Op 分发

- `include/vt/ops.h` 定义 `OpId` 枚举（~100+ 个 op：Matmul、RmsNorm、Rope、GdnPrefill、PagedAttention、Sampler 系列等）。
- `src/vt/op_provider.cpp` 维护加速提供者注册表；`GetOp(OpId, DeviceType)` 返回对应函数指针。
- `src/vt/ops.cpp` 每个 wrapper 做 shape/dtype/device 校验后调用 `GetOp`。
- 设计目标是：同一 op 在 CPU 上有参考实现，在 CUDA 上有调优内核，结果 byte-identical（在 greedy 语义下）。

### 7.3 后端实现

```text
src/vt/cuda/     → cuBLASLt / CUTLASS / Triton-AOT cubins / 手写 CUDA kernels
src/vt/cpu/      → C++ 参考内核，用于 CI、op parity、无 GPU 机器
src/vt/metal/    → Apple Silicon Metal / 可选 MLX GEMM provider
src/vt/vulkan/   → 可移植 GPU 骨架（当前 8 个 op + fusion catalog 校验）
```

CUDA 侧把 Triton-AOT cubin（GDN、NVFP4 等）vendor 进仓库，编译运行不需要 Python/Triton。

---

## 8. 模型注册与加载

### 8.1 Model Registry

`include/vllm/model_executor/models/model_registry.h`：

- `ModelInfo`：is_text_generation / is_pooling / is_hybrid / supports_multimodal。
- `ModelRegistration`：架构名 → 工厂函数 + 权重加载器。
- `LoadedModel`：类型擦除的已加载模型，runner 通过 `ModelForwardInput` 调用其 `Forward` 钩子。

### 8.2 加载路径

```text
FromModelDir(path)
    → 识别 safetensors 或 GGUF
    → HfConfig / GGUF metadata 解析
    → 按 architecture 选择 ModelRegistration
    → 加载权重（可能保持量化格式：GGUF keep-quant、NVFP4 W4A4、FP8 等）
    → 构建 KVCacheConfig、Scheduler、Runner、EngineCore、LLMEngine/AsyncLLM
```

### 8.3 量化支持

| 类型 | 说明 |
|------|------|
| NVFP4 W4A4 / W4A16 | MoE grouped GEMM、dense GEMM，权重保持 fp4 在显存 |
| FP8 W8A8 | 部分路径支持 |
| GGUF | F32/F16/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K，CPU 直接计算压缩块 |
| Marlin | W4A16 等交错布局 |

### 8.4 GGUF 支持详情

GGUF 是 vllm.cpp 的一等公民输入格式，与 safetensors 并列：

- **C ABI**：`vllm_model_params.model_path` 可直接指向 `.gguf` 文件（`include/vllm.h:127`）。
- **加载器**：`ModelSource::FromGguf(...)` 与 `FromSafetensors` 并列注册到 `ModelRegistry`（`include/vllm/model_executor/models/model_registry.h:75`）。
- **tokenizer**：GGUF 内置的 `tokenizer.chat_template` 元数据会被用于 chat 入口；也可用 `--tokenizer-config` 覆盖。
- **量化计算**：
  - CPU 路径直接对压缩块做 GEMM/GEMV，无需反量化为 BF16，内存占用与 llama.cpp 同级。
  - CUDA 路径对 keep-quant 权重通过 `vt::MatmulBTQuant` / `vt::MatmulBTQuantGrouped` 计算。
- **已验证模型**：Qwen3.6-35B-A3B（GGUF 35B）、DeepSeek-V4-Flash（GGUF 80.7 GB）等。
- **MVP 门控**：`AGENTS.md` 把 “loading from safetensors **and GGUF**” 列为 MVP gate 之一。

---

## 9. 高级特性管线

### 9.1 投机解码（Speculative Decoding）

`vllm/v1/spec_decode/`、`vllm/v1/worker/gpu/spec_decode/`：

- **MTP**：Qwen3.5/3.6 自带 `mtp.*` 头，verify/propose 循环。
- **DFlash**：block-diffusion 草稿模型。
- **ngram**：draft-free ngram 提议器。
- 拒绝采样（`rejection_sampler.cpp`）保证与 vLLM 同样 token 一致。

### 9.2 结构化输出

`vllm/v1/structured_output/`：

- 支持 JSON schema、regex、choice、GBNF grammar、JSON mode。
- 引擎每步生成 grammar bitmask，采样前把禁止 token 的 logit 置 `-inf`。
- xgrammar C++ core 作为默认后端，native / 其他后端可插拔。
- SGLang jump-forward decoding 可选加速（仅 token-unique 强制路径）。

### 9.3 Tool Calling 与 Reasoning

- 36 个 tool-parser 家族、9 个 reasoning parser。
- chat template 引擎使用 vendored `google/minja`（与 llama.cpp 相同）。
- 引擎侧流式解析 tool_calls / reasoning content。

### 9.4 多模态

- image / video / audio 路径通过 `multimodal::MultiModalInputs` 进入引擎。
- vision tower 在主机侧（或 encoder runner）预处理，生成 merged embeddings 与 MRoPE positions。
- 已支持 Qwen3-VL、Qwen3.6-27B vision、Voxtral audio。

### 9.5 外部 KV 卸载

`vllm/v1/kv_offload/`：

- CPU/disk tiering、LMCache `lm://` 客户端。
- 调度器侧 connector + runner 侧 store/load，opt-in，关闭时行为完全一致。

---

## 10. 测试与正确性门

项目把 vLLM 当作 oracle，要求：

1. **Op parity**：与上游 golden dump 在 CPU + CUDA 上逐项对齐。
2. **Engine behavioral**：调度、KV 分配、抢占、前缀缓存等行为测试。
3. **Model parity**：logits + greedy token-for-token，门控模型 Qwen3.6-27B / 35B-A3B。
4. **Server e2e**：OpenAI API、tool-call streaming、grammar 约束。
5. **Gate benchmark**：与 vLLM `vllm bench throughput` 同 workload 对比，要求 match 或 beat。

---

## 11. 数据流总结

一次完整的 `/v1/chat/completions` 请求在引擎内部的数据流：

```text
HTTP 请求
  → OpenAI serving_chat：解析 messages/tools/response_format
  → chat template + tool/reasoning 解析器
  → InputProcessor 分词 → EngineCoreRequest
  → AsyncLLM/LLMEngine 注册输出收集器
  → EngineCore.add_request() → Scheduler.waiting 队列

每步循环
  → Scheduler.schedule() 产出 SchedulerOutput
  → GPUModelRunner.update_states() 同步 InputBatch
  → decode-first reorder + prepare_inputs()
  → 注册模型 Forward(token_ids, positions, attn_meta, gdn_meta, kv_caches)
      → vt:: 各 op 在选定后端执行
  → 收集 [num_reqs, vocab] logits
  → apply_grammar_bitmask()
  → Sampler::forward() 产出 sampled token ids
  → Scheduler.update_from_output()：追加 token、检查 stop、释放 KV
  → OutputProcessor：detokenize、组装 delta、tool_calls/reasoning 切片
  → 通过 collector 返回给 HTTP SSE / 阻塞调用
```

---

## 12. 关键源码速查

| 层级 | 关键文件 |
|------|----------|
| C ABI | `include/vllm.h` |
| 同步引擎 | `include/vllm/v1/engine/llm_engine.h`、`src/vllm/v1/engine/llm_engine.cpp` |
| 异步引擎 | `include/vllm/v1/engine/async_llm.h` |
| EngineCore | `include/vllm/v1/engine/core.h`、`src/vllm/v1/engine/core.cpp` |
| 调度器 | `include/vllm/v1/core/sched/scheduler.h`、`src/vllm/v1/core/sched/scheduler.cpp` |
| KV Cache | `include/vllm/v1/core/kv_cache_manager.h`、`include/vllm/v1/core/block_pool.h` |
| Runner | `include/vllm/v1/worker/gpu/runner.h`、`src/vllm/v1/worker/gpu/runner.cpp` |
| InputBatch | `include/vllm/v1/worker/gpu/input_batch.h` |
| 注意力/GDN | `include/vllm/v1/attention/backends/gdn_attn.h` |
| 模型注册 | `include/vllm/model_executor/models/model_registry.h` |
| vt Tensor | `include/vt/tensor.h`、`include/vt/backend.h`、`include/vt/ops.h` |
| vt op 分发 | `src/vt/op_provider.cpp`、`src/vt/ops.cpp` |
| 服务端 | `examples/server/main.cpp`、`vllm/entrypoints/openai/api_server.h` |
| CLI | `examples/cli/main.cpp` |

---

*文档由 Kimi Code 根据当前仓库源码与 AGENTS.md / README.md / docs 生成。*
