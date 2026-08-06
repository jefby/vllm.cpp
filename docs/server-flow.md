# vllm.cpp OpenAI Server 流程详解

> 对应代码：`examples/server/main.cpp`、`vllm/entrypoints/openai/*`、`vllm/entrypoints/model_loader.*`、`vllm/v1/engine/async_llm.*`。

---

## 1. 总体流程概览

```text
用户 HTTP 请求
    → cpp-httplib (examples/server → ApiServer)
        → 路由分发：/v1/completions、/v1/chat/completions、/v1/models ...
            → OpenAIServingCompletion / OpenAIServingChat
                → 协议解析 → SamplingParams / ChatPromptFn / ToolParser / ReasoningParser
                    → AsyncLLM.add_request() → EngineCore 调度 → GPUModelRunner 执行
                ← 收集 RequestOutput（同步或 SSE 流式）
            ← 组装 OpenAI 格式响应 / SSE chunk
    ← HTTP 响应
```

核心设计：**HTTP 传输层薄，协议逻辑在 serving handler，引擎执行在 AsyncLLM**。这样单元测试可以不绑端口直接测 `ApiServer::handle_*`。

---

## 2. 启动阶段

### 2.1 `examples/server/main.cpp`

1. **解析命令行**：`--model`、`--host`、`--port`、`--max-num-seqs`、`--max-num-batched-tokens`、`--enable-prefix-caching`、`--scheduling-policy`、`--tool-call-parser`、`--reasoning-parser`、`--kv-transfer-config`、`--speculative-config` 等。
2. **加载模型**：
   ```text
   EngineParams params = {...};
   auto engine = LoadedEngine::FromModelDir(model_dir, params);
   ```
3. **构造 serving 对象**：
   - `OpenAIServingCompletion`：使用 `engine->async_engine()`。
   - `OpenAIServingChat`：使用 `async_engine()` + `MakeChatTemplatePromptFn(...)` + tool/reasoning parser。
   - `OpenAIServingModels`：模型名检查。
4. **构造 `ApiServer`**：绑定 completion / chat / models handler，配置最大并发流。
5. **注册可选端点**：`/metrics`、`/tokenize`、`/detokenize`、`/reset_prefix_cache`、`/abort_requests`、`/tokenizer_info`。
6. **listen(host, port)**：阻塞运行 HTTP 服务。

### 2.2 `LoadedEngine::FromModelDir`

```text
模型目录 / GGUF 文件
    → LoadHfConfig / GGUF metadata
    → Load tokenizer (HF tokenizer.json 或 GGUF vocab)
    → ModelRegistry 选择对应 architecture
    → 加载权重（safetensors 或 GGUF）
    → SelectQueue() 按 CurrentPlatform() 选择 CUDA / Metal / Vulkan / CPU
    → 构造 KVCacheConfig、Scheduler、Executor、EngineCore
    → 构造 LLMEngine
    → （按需）AsyncLLM 懒启动
```

`LoadedEngine` 持有完整引擎栈；`async_engine()` 在第一次调用时启动 `EngineCoreProc` 线程 + 输出处理线程。

---

## 3. HTTP 传输层：`ApiServer`

### 3.1 实现：cpp-httplib

- 依赖 `third_party/httplib/httplib.h`，header-only MIT 库（与 llama.cpp server 同款）。
- CMake 选项 `VLLM_CPP_SERVER` 控制是否编译；缺少头文件时自动关闭。
- 启用 `TCP_NODELAY`，让 SSE 逐 token 帧立即发出，避免 Nagle 合并。

### 3.2 Worker Pool

```cpp
static constexpr size_t kDefaultMaxConcurrentStreams = 8;
static constexpr size_t kControlWorkerHeadroom = 4;
```

固定线程池大小 = `max_concurrent_streams + 4`。每个 SSE 流会占一个 worker 等待其 collector；预留 4 个给 health / models / metrics 等控制面请求。

### 3.3 路由表

| 方法 | 路径 | Handler |
|------|------|---------|
| POST | `/v1/completions` | `handle_completions` |
| POST | `/v1/chat/completions` | `handle_chat_completions` |
| GET  | `/v1/models` | `handle_models` |
| GET  | `/health`、`/ping` | `handle_health` / `handle_ping` |
| GET  | `/version` | `handle_version` |
| GET  | `/metrics` | `handle_metrics`（需 `set_metrics_logger`） |
| POST | `/tokenize`、`/detokenize` | `handle_tokenize` / `handle_detokenize`（需 tokenizer） |
| POST | `/reset_prefix_cache` | `handle_reset_prefix_cache`（需回调） |
| POST | `/abort_requests` | `handle_abort_requests`（需回调，dev mode） |
| GET  | `/tokenizer_info` | `handle_tokenizer_info`（需 enable flag） |

---

## 4. `/v1/completions` 详细流程

### 4.1 请求解析

```text
HTTP body (JSON)
    → nlohmann::json::parse
    → CompletionRequest (from_json)
        → prompt / model / max_tokens / temperature / top_p / top_k / stream / ...
```

### 4.2 模型校验

```cpp
if (!models_.check_model(request.model)) return 404;
```

### 4.3 生成处理

```text
OpenAIServingCompletion::create_completion(request)
    → request_id = "cmpl-{counter}"
    → 如果 request.use_beam_search:
         → BeamSearchAsync / BeamSearch（注意：stream 不支持 beam search）
    → 否则:
         → SamplingParams sampling_params = request.to_sampling_params();
         → engine_request_id = request_id + "-0";
         → if async_engine && stream:
                async_request = async_engine_->add_request(engine_request_id, prompt,
                                                           sampling_params, priority);
                return CompletionSseStream(async_engine, async_request, ...);
           else if stream:
                sync_engine_->add_request(...);
                while (has_unfinished) { step(); collect deltas; }
                return precomputed sse_chunks;
           else:
                sync_engine_->generate(prompt, sampling_params);
                return CompletionResponse;
```

### 4.4 响应格式

- **非流式**：`CompletionResponse`（id / created / model / choices / usage）。
- **流式**：SSE 帧 `data: {...}\n\n`，最后 `data: [DONE]\n\n`。

---

## 5. `/v1/chat/completions` 详细流程

这是服务端最复杂的端点，涉及 chat template、多模态、tool/reasoning 解析、结构化输出。

### 5.1 请求解析

```text
ChatCompletionRequest
    → messages[]
    → model / temperature / max_tokens / stream / stream_options
    → tools[] + tool_choice
    → response_format（structured output）
    → include_reasoning
```

### 5.2 Prompt 构造

```text
OpenAIServingChat::create_chat_completion
    → 如果消息包含 image/audio 等多模态 part:
         → MultiModalChatFn 生成 MultiModalInputs（placeholder 扩展的 prompt ids + mm_features）
    → 否则:
         → ChatPromptFn(messages, add_generation_prompt, tools) 渲染 prompt 字符串
         → 默认回退："<role>: <content>\n" 拼接 + "assistant:"
         → 生产环境：MakeChatTemplatePromptFn 使用模型 tokenizer_config.json 中的 Jinja 模板
```

### 5.3 结构化输出与 Tool Choice

- `response_format` 为 JSON schema / regex / grammar 等时，写入 `SamplingParams.structured_outputs`。
- `tool_choice` 为 `auto` / `required` / 指定 function 时，通过 `ApplyToolChoiceStructuredOutput` 给采样加 structural-tag 约束（如 Hermes 格式）。

### 5.4 提交引擎

```text
if multimodal:
    async_engine_->add_request(engine_request_id, mm_inputs, sampling_params, priority);
else:
    async_engine_->add_request(engine_request_id, prompt, sampling_params, priority);
```

### 5.5 非流式后处理

```text
model_output = engine generate 结果
    → ReasoningParser 先提取 reasoning content（如 <think>...</think>）
    → ToolParser 在剩余 content 上提取 tool_calls
    → ShapeChatMessage / ShapeChatMessageEngine
    → finish_reason = "stop" | "length" | "tool_calls"
    → 组装 ChatCompletionResponse
```

### 5.6 流式后处理

```text
ChatSseStream::next()
    → 从 AsyncLLM collector 拉取 RequestOutput delta
    → 对每个 delta_text:
         → ReasoningParser.extract_reasoning_streaming（分离 reasoning 与 content）
         → ToolParser.extract_tool_calls_streaming（状态化解析 tool_calls）
         → ShapeChatDelta / ShapeChatDeltaEngine
    → 输出 SSE chunk: {role / delta.content / delta.tool_calls / delta.reasoning / finish_reason}
    → 最后输出 usage（可选）和 [DONE]
```

流式 tool calling 是**状态化**的：parser 累积文本，在识别到完整 tool call 时逐步发出 `tool_calls` delta。

---

## 6. 其他端点

### 6.1 `/v1/models`

`OpenAIServingModels::show_available_models()` 返回 `{object:"list", data:[{id, object, created, owned_by}]}`。

### 6.2 `/metrics`

- 依赖 `PrometheusStatLogger`。
- 返回 Prometheus text exposition 格式，指标名与 vLLM 一致。

### 6.3 `/tokenize` / `/detokenize`

- `/tokenize` 支持两种形式：
  - `{prompt: "..."}` → 直接 tokenize。
  - `{messages: [...]}` → 先走 chat template 渲染，再 tokenize。
- `/detokenize` 将 token id 数组还原为字符串。

### 6.4 `/reset_prefix_cache`

调用 `EngineCore::reset_prefix_cache(reset_running_requests, reset_external)`，清空前缀缓存块。

### 6.5 `/abort_requests`

Dev mode 端点，解析 `{request_ids: [...]}`，调用 `AsyncLLM::abort()`。空列表表示 abort 所有 in-flight 请求。

---

## 7. Streaming / SSE 实现

两个流式实现共享 `SseStream` 接口：

```cpp
struct SseStream {
    virtual bool next(std::string& chunk) = 0;  // 返回 false 表示结束
    virtual void abort() = 0;
};
```

- **Legacy sync 路径**：`sync_engine_` 驱动 `step()` 循环，把所有 chunk 预计算到 `sse_chunks` vector 中返回。测试/兼容路径。
- **AsyncLLM  live 路径**：`CompletionSseStream` / `ChatSseStream` 持有 `AsyncRequest`，每次 `next()` 阻塞在 `AsyncLLM::get_output()` 上，实时拉取最新 delta。HTTP worker 线程在此期间阻塞等待，不会占用引擎线程。

`ApiServer::listen()` 中：
- 如果 `sse_stream` 存在，注册一个长轮询 handler，反复调用 `next()` 直到结束或客户端断开。
- 断开时调用 `sse_stream->abort()`，通知引擎终止该请求。

---

## 8. 同步 vs 异步 Engine 模式

| 模式 | 使用场景 | 特点 |
|------|---------|------|
| `LLMEngine`（同步） | `examples/cli`、legacy server 路径、单元测试 | 单请求 `generate()` 阻塞 |
| `AsyncLLM`（异步） | `examples/server` 生产路径 | 多请求并发，输出处理线程，batch-queue depth=2 重叠前向与采样输出 copy |

`OpenAIServingCompletion/Chat` 同时支持两种构造：

```cpp
OpenAIServingChat(v1::LLMEngine& engine, ...);   // 同步
OpenAIServingChat(v1::AsyncLLM& engine, ...);    // 异步
```

生产 server 走 `AsyncLLM`；legacy_engine_mutex 只在同步模式下加锁。

---

## 9. 错误处理

统一 `ErrorResponse` JSON 格式：

- **400 BadRequestError**：JSON 解析失败、字段非法。
- **404 NotFoundError**：请求的 model 不存在。
- **500 InternalServerError**：引擎异常，会打印 `api-server: 500 endpoint=... model=... what=...` 到 stderr 方便定位。

流式请求发生 500 时，sse stream 析构会自动 `abort()` 对应引擎请求，避免残留。

---

## 10. 关键源码索引

| 组件 | 文件 |
|------|------|
| Server 入口 | `examples/server/main.cpp` |
| HTTP 路由与传输 | `src/vllm/entrypoints/openai/api_server.cpp`、`include/vllm/entrypoints/openai/api_server.h` |
| `/v1/completions` 逻辑 | `src/vllm/entrypoints/openai/serving_completion.cpp`、`include/vllm/entrypoints/openai/serving_completion.h` |
| `/v1/chat/completions` 逻辑 | `src/vllm/entrypoints/openai/serving_chat.cpp`、`include/vllm/entrypoints/openai/serving_chat.h` |
| OpenAI 协议类型 | `include/vllm/entrypoints/openai/protocol.h`、`src/vllm/entrypoints/openai/protocol.cpp` |
| 模型加载与引擎栈 | `src/vllm/entrypoints/model_loader.cpp`、`include/vllm/entrypoints/model_loader.h` |
| Chat Template | `src/vllm/entrypoints/chat_template.*` |
| 多模态 chat 预处理 | `src/vllm/entrypoints/openai/chat_mm.*` |
| Tool / Reasoning Parser | `src/vllm/entrypoints/openai/tool_parsers/*`、`src/vllm/entrypoints/openai/reasoning_parsers/*` |
| 异步引擎 | `src/vllm/v1/engine/async_llm.cpp`、`include/vllm/v1/engine/async_llm.h` |
| SSE 接口 | `include/vllm/entrypoints/openai/serving_completion.h`（`SseStream`） |

---

*文档由 Kimi Code 根据当前仓库源码生成。*
