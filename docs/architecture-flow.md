# vllm.cpp 软件架构流程分析

> 生成时间：基于合并 `main` 后的最新代码快照（2026-08-11）。
> 分析范围：C++ 推理引擎核心、vt 张量运行时、调度与执行管线、服务入口、C ABI 与多模态扩展。

---

## 1. 总体定位与设计哲学

`vllm.cpp` 是 vLLM V1 / Model Runner V2 的纯 C++ 移植目标：

- **无 Python / 无 PyTorch / 无 ggml 依赖**：推理时不带解释器，二进制仅 ~66 MiB。
- **行为镜像 vLLM**：同一 workload、同一模型、greedy 解码要求 token-for-token 一致。
- **llama.cpp 式部署**：核心产物是 `libvllm` + 稳定 C ABI（`include/vllm.h`），上层再包 CLI / OpenAI 服务器。
- **GGUF 一等公民**：与 safetensors 并列支持，CPU 可直接在量化块上计算。
- **多后端同构**：CPU、CUDA、Metal、ROCm、Vulkan、Tenstorrent 已通过统一 `vt::` 运行时接入，XPU / ANE 后续跟进，引擎代码不感知具体硬件。

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

`include/vllm.h` 是稳定的 C 接口，当前 ABI 版本 `VLLM_ABI_VERSION 17`：

- `vllm_engine_load()`：从模型目录或 **GGUF 文件**构建完整引擎（`model_path` 直接接受 `.gguf`）。
- `vllm_complete()` / `vllm_complete_stream()`：阻塞式或流式完成。
- `vllm_complete_tokens()`（ABI v13）：预 tokenized 的阻塞式完成。
- `vllm_request_submit()` / `vllm_request_wait()`：非阻塞异步请求。
- `vllm_chat()` / `vllm_chat_stream()`：OpenAI 风格 chat，引擎侧做 chat template、tool/reasoning 解析。
- `vllm_transcribe()` / `vllm_transcription_params`（ABI v11）：Parakeet 语音转文字。
- `vllm_embed()` / `vllm_embedding_result`（ABI v15）：Llama 等 pooling 模型获取文本嵌入。
- `vllm_video_engine_load()` / `vllm_video_generate()` / `vllm_video_mux_argv()`（ABI v12）：MiniMax-H3 视频+音频生成与 mux。
- `vllm_server_main()`：以库形式直接启动 OpenAI 兼容 HTTP 服务器（`examples/server` 是其薄封装）。
- `vllm_abi_version()`：返回编译时 ABI 版本。
- 结构化输出、投机解码、前缀缓存、调度策略、外部 KV 传输均通过 `vllm_model_params` 字段配置。

### 3.2 CLI

`examples/cli/main.cpp` 完全走 C ABI，并暴露 `--max-num-seqs` 等关键调度参数：

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
| POST | `/v1/embeddings` | 文本嵌入（pooling 模型） |
| POST | `/v1/audio/transcriptions` | 语音转文字（Parakeet） |
| POST | `/v1/videos` | 视频生成任务入队（MiniMax-H3） |
| POST | `/v1/videos/sync` | 同步视频生成，返回 MP4 |
| GET  | `/v1/videos/{id}` | 查询异步视频任务状态 |
| GET  | `/v1/videos/{id}/content` | 获取已完成 MP4 字节 |
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

> 完整 server 流程（启动、路由、/v1/completions、/v1/chat/completions、SSE、错误处理）见 [docs/server-flow.md](server-flow.md)。

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

> 注：经典 Dense 模型（如 Qwen3 系列）的异步服务路径已默认使用设备镜像（Async Dense Mirror），将 `Qwen3ForCausalLM` 的同步 `LLMEngine` 执行迁移到 `AsyncLLM`，消除同步 server 的 GPU 空闲。
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
       - Llama / Mistral / Gemma-4 / DeepSeek-V4 / Kimi-Linear-48B / Muse-Glimmer / MiniMax-H3 / Parakeet / ... 等注册模型
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
src/vt/cuda/         → cuBLASLt / CUTLASS / Triton-AOT cubins / 手写 CUDA kernels
src/vt/cpu/          → C++ 参考内核，用于 CI、op parity、无 GPU 机器
src/vt/metal/        → Apple Silicon Metal / 可选 MLX GEMM provider
src/vt/rocm/         → AMD ROCm HIP 后端（HIPBLASLt、自定义 kernel）
src/vt/vulkan/       → 可移植 GPU 后端（compute shaders + SPIR-V）
src/vt/tenstorrent/  → Tenstorrent 设备后端（TTNN / mesh-trace）
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
- 视频生成：MiniMax-H3 提供 `vllm_video_*` C API 与 `/v1/videos` 端点，生成帧 + WAV 后由调用者 exec ffmpeg 得到 MP4。
- 语音转写：Parakeet CTC/RNN-T/TDT 家族通过 `vllm_transcribe` 与 `/v1/audio/transcriptions` 提供服务。
- 文本嵌入：Llama 等 pooling 模型通过 `vllm_embed` 与 `/v1/embeddings` 提供服务。
- 已支持 Qwen3-VL、Qwen3.6-27B vision、Voxtral audio、Muse-Glimmer、MiniMax-H3、Parakeet。

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

## 13. 模型支持示例：Qwen3.6-35B-A3B

Qwen3.6-35B-A3B 是当前代码的一等公民目标模型之一，MoE + GDN-hybrid 架构，支持 safetensors 与 GGUF 双路径加载。

### 13.1 架构注册

- 注册名：`"Qwen3_5MoeForConditionalGeneration"`（`src/vllm/model_executor/models/qwen3_5_moe.cpp:178`）。
- 该 TU 仅负责 registry 粘合；核心前向在 `src/vllm/model_executor/models/qwen3_5.cpp`（`Qwen3_5Model::Forward` / `ForwardDevice` / `Qwen3_5DecodeGraph`）。
- 同步密集版本另有 `"Qwen3_5ForConditionalGeneration"`（`src/vllm/model_executor/models/qwen3_5_dense.cpp:208`）和经典 `"Qwen3ForCausalLM"`（`src/vllm/model_executor/models/qwen3_dense.cpp:155`）。

### 13.2 加载路径

| 格式 | 入口 | 说明 |
|------|------|------|
| Safetensors NVFP4 | `LoadQwen3_5Moe` in `qwen3_5_weights.cpp` | nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot `491c2f1e` 是测试基准 |
| GGUF | `LoadQwen3_5MoeFromGguf` in `qwen3_5_gguf_weights.cpp` | 支持 F32/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 等混合量化 |
| C ABI | `vllm_engine_load()` | `model_path` 直接指向模型目录或 `.gguf` 文件 |

### 13.3 已支持的高级特性

- **NVFP4 W4A4/W4A16**：CUDA Blackwell 路径默认 keep-quant，权重以 FP4 驻留显存；Marlin 驻留与 grouped GEMM 可选。
- **GGUF keep-quant**：CPU / CUDA 直接对压缩块计算，无需整体反量化为 BF16。
- **MTP 投机解码**：35B 自带 `mtp.*` 草稿头可复用本模型的 embed / lm_head。
- **DFlash / DSpark spec**：MoE 前向支持 `aux_tap` 多 tap 输出，用于 block-diffusion / dspark verify。
- **前缀缓存**：dense full-attention 层默认开启；GDN 组默认关闭。
- **异步设备镜像**：35B 的 dense 变体已默认走 `AsyncLLM` 消除同步 server 的 GPU 空闲。

### 13.4 性能基准

`docs/BENCHMARKS.md` 中 Qwen3.6-35B-A3B NVFP4 `nvidia` vs vLLM 0.25.0（GB10）：

| 并发 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| vllm.cpp / vLLM 比率 | 0.918x | 0.940x | 0.972x | 0.956x | 0.93x | — |
| 状态 | near-tie | near-tie | near-tie | near-tie | open gap | 未测量 |

> 注：c16 0.93x 是仍未关闭的 open gap；项目目标是 match or beat vLLM on every axis。

---

## 14. 与 llama.cpp 的速度对比

vllm.cpp 的**主要性能目标是 vLLM parity**，llama.cpp 仅作为 **secondary floor**（当 vLLM 无法加载同一 checkpoint 时使用）。因此对比结果高度依赖后端、模型和比较维度。

### 14.1 已测量数据（来源：`docs/BENCHMARKS.md`、`.agents/benchmark-record.md`）

| 场景 | 后端/量化 | vllm.cpp | llama.cpp | 比率 | 解读 |
|---|---|---:|---:|---:|---|
| 20-core ARM i8mm (GB10 aarch64) | CPU GGUF | prefill 223.8 / decode 24.7 tok/s | prefill 177.3 / decode 25.4 | prefill **1.18x** / decode 0.97x（tie） | 服务器级 ARM 上 vllm.cpp prefill 领先，decode 持平，内存持平 |
| RPi5 A76 4-core | CPU GGUF | prefill 12.81 / decode 2.55 | prefill 27.77 / decode 3.91 | prefill **0.461x** / decode **0.653x** | 低功耗 ARM 上 llama.cpp 仍更快；vllm.cpp RSS 少 24.2% |
| GB10 Vulkan | Vulkan GGUF 27B | prefill 21.5x | baseline | prefill **21.5x** | Vulkan 路径 prefill 大幅领先，decode 4.36 vs 4.35 持平 |
| Muse Glimmer 30B | CPU GGUF | prefill 0.997x / decode 0.232x | baseline | prefill tie / decode **0.232x** | 新模型 decode 仍显著落后于 llama.cpp |
| Laguna-S-2.1 batch-1 | CPU GGUF Q4_K_XL | 7.7 tok/s | 27.8 tok/s | 0.28x | 单流大 MoE GGUF 上 llama.cpp 的 GEMV 仍是 best-in-class；但 vllm.cpp 的 NVFP4 CUDA 路径 vs vLLM 是 near-tie |

### 14.2 关键结论

1. **CUDA 路径**：vllm.cpp 不直接 vs llama.cpp，因为 llama.cpp 通常无法加载 NVFP4/FP8 safetensors；vllm.cpp 与 vLLM 对比（Qwen3.6-27B 1.045x–1.017x c1–c32）是主战场。
2. **服务器级 ARM CPU**：vllm.cpp 凭借 assembly/SDOT 优化可在 prefill 上击败 llama.cpp，decode 持平。
3. **低功耗 ARM / 新模型 GGUF decode**：llama.cpp 的 GGUF 手写 GEMV 更成熟，vllm.cpp 仍有 gap。
4. **Vulkan 后端**：vllm.cpp 在 GB10 上 prefill 大幅领先，decode 持平。
5. **内存**：vllm.cpp 在 Pi 上 RSS 少 24.2%；在 GB10 aarch64 上几乎持平。

---

## 15. Qwen3.6-27B 支持速查

Qwen3.6-27B 是 vllm.cpp 的 MVP gate 模型之一，走 **dense / GDN-hybrid** 路径，与 35B-A3B 共享同一个 `Qwen3_5Model` 前向骨架，但**无 MoE**。

### 15.1 架构注册

- 注册名：`"Qwen3_5ForConditionalGeneration"`（`src/vllm/model_executor/models/qwen3_5_dense.cpp:208`）。
- 文件注释明确写 “DENSE Qwen3.6-27B text gate”。
- 核心前向同样在 `src/vllm/model_executor/models/qwen3_5.cpp`（`Qwen3_5Model::Forward` / `ForwardDevice` / `Qwen3_5DenseDecodeGraph`）。
- 经典 Dense 版本另有 `"Qwen3ForCausalLM"`（`src/vllm/model_executor/models/qwen3_dense.cpp:155`）。

### 15.2 加载路径

| 格式 | 入口 | 说明 |
|------|------|------|
| Safetensors NVFP4 | `LoadQwen3_5Dense` in `qwen3_5_dense_weights.cpp` | 同时覆盖 `unsloth/Qwen3.6-27B-NVFP4` 与 `nvidia/Qwen3.6-27B-NVFP4` 两个发布源 |
| GGUF | `LoadQwen3_5DenseFromGguf` in `qwen3_5_gguf_weights.cpp` | 支持 Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / F32 等混合量化，CPU/CUDA keep-quant 直接算 |
| C ABI | `vllm_engine_load()` | `model_path` 直接指向模型目录或 `.gguf` 文件 |

### 15.3 已支持的高级特性

- **NVFP4 W4A4 / W4A16**：CUDA Blackwell 路径默认 keep-quant；`lm_head` FP4 打包、`GDN FP8 QKVZ` 合并优化已落地。
- **GGUF keep-quant**：CPU / CUDA 直接对压缩块计算，无需整体反量化为 BF16。
- **CUDA decode graph**：`Qwen3_5DenseDecodeGraph`（`qwen3_5_dense.cpp` 中启用）。
- **MTP 投机解码**：`qwen3_5_mtp.h` 提供 27B / 35B 共享的 MTP 草稿层。
- **DFlash / DSpark spec**：`qwen3_dflash.h` / `qwen3_dspark.h` 针对 27B 提供 speculative verify 路径。
- **前缀缓存**：dense full-attention 层默认开启；GDN 组默认关闭。
- **异步设备镜像**：dense Qwen3 路径已默认迁移到 `AsyncLLM`。

### 15.4 测试与性能基准

- Parity golden：`tests/parity/goldens/qwen36_*_27b/` 包含 logits、GDN layer、full-attention layer、norm、embed、MTP head 等测试基准。
- 功能测试：`tests/vllm/models/test_qwen27_paged_forward.cpp`、`tests/parity/test_qwen27_gguf_nvfp4_compute.cpp` 等。
- `docs/BENCHMARKS.md` 中 Qwen3.6-27B 的绑定对比（GB10）：

| Checkpoint | vs vLLM 0.25.0 | 结果 |
|---|---|---|
| `unsloth/Qwen3.6-27B-NVFP4` @`890bdef7` | c1 1.045x，c2–c32 1.011–1.017x | **ahead / tie** |
| `nvidia/Qwen3.6-27B-NVFP4` @`0893e160` (ModelOpt `modelopt_mixed`) | c1 0.838x，c2–c8 0.964–0.967x | **behind / near-tie** |

> 两个 27B 同名 checkpoint 因量化分布不同（unsloth vs nvidia ModelOpt）被当作不同行对比。

---

*文档由 Kimi Code 根据当前仓库源码与 AGENTS.md / README.md / docs 生成。*
