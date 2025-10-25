# SceneImg 数据采集与筛选工具集

SceneImg 提供一套面向机器人仿真/任务生成的室内图像采集、筛选与去重流水线。通过百度关键词抓取、以图搜图拓展、CLIP 初筛和 VLM 精筛的多阶段流程，构建具有真实生活气
息、适度杂乱、可操作物体丰富的图像集合。

## 目录结构概览

- `crawler.py`：百度关键词异步爬虫（全局去重、元数据记录）。
- `search_by_images.py`：基于种子图片的以图搜图扩展（真分页、dHash 相似度过滤）。
- `filter_basic.py`：CLIP + 可选检测器的初筛，筛出室内且物体充足的候选图像。
- `filter_vlm.py`：多模态大模型精筛，生成评分、原因、标签并按需拷贝结果。
- `util/index_existing.py`：根据现有图片重建 `_global_index.jsonl` 或 `_filtered_index.jsonl`。
- `util/remove_seed_duplicates.py`：删除与种子几乎相同的下载图片。
- `prompt/`：VLM 过滤提示词（多版本演进）。
- `docs/summary.md`：现有流程总结与改进建议。
- `crawledimg/`, `filteredimg/`：默认输出目录。
- `testimg/`, `crawledimg/search_by_images/seed/`：示例/种子图片目录。

## 快速开始

1. **准备环境**
    - Python 3.10+，推荐创建虚拟环境。
    - 安装依赖：`pip install -r requirements.txt`（若无，可按脚本内导入手动安装 `aiohttp`, `Pillow`, `torch`, `torchvision`, `clip`, `tqdm`, `requests` 等）。
    - 若使用 OpenAI，需要配置 `OPENAI_API_KEY`；若使用 Ollama 云端，配置 `OLLAMA_API_KEY`。

2. **目录准备**
    - 创建输出目录（脚本会按需建立子目录）。
    - 将种子图片放入 `crawledimg/search_by_images/seed` 或自定义目录。

3. **运行顺序建议**
    1. `crawler.py` 根据关键词抓取基础图片。
    2. `search_by_images.py` 以图搜图拓展长尾场景。
    3. `filter_basic.py` 做 CLIP 初筛（可启用物体数量约束）。
    4. `filter_vlm.py` 进行多模态模型精筛，生成最终候选集合。

---

## 关键脚本详解

### crawler.py —— 百度关键词异步爬虫

#### 实现思路
- 使用 `aiohttp` 并发请求百度图片接口 `acjson`，先访问搜索页 `search/index` 完成 cookie 预热。
- `GlobalDedupIndex` 持久化跨运行的 URL MD5 与图片 SHA1，避免重复下载。
- 下载后对图片进行 `Pillow` 归一化：自动纠正 EXIF 方向、按内容选择 JPEG/PNG，写入同名 `.json` 元数据文件（含来源 URL、尺寸等）。
- 并发控制：`TCPConnector.limit_per_host` 限制每主机连接数，通过 `batch_size` 批量 gather。

#### 使用方法
```bash
python crawler.py \
--default-queries \
--limit-per-query 200 \
--output crawledimg/baidu/raw \
--timestamp-subdir \
--per-host 8 \
--batch-size 20 \
--debug
```

参数说明：

- --query/-q、--query-file：自定义关键词或文件。
- --default-queries：使用脚本内置的中文关键词。
- --timestamp-subdir：每次运行在查询目录下创建时间戳子目录。
- --debug：打印调试信息，并（如需要）放宽尺寸/宽高比过滤。

———

### search_by_images.py —— 以图搜图拓展

#### 实现思路

- 将种子图片上传至 graph.baidu.com/upload 获得查询页面/签名，支持新版 Base64 __PRELOADED_STATE__ 解析。
- 解析主页面、AJAX 接口与 iframe，获取真实分页参数，循环抓取类似图片。
- 下载流程沿用 GlobalDedupIndex 与归一化逻辑，同时可选启用 dHash 距离过滤（去除与种子过近/过远的图片）。
- 为每个种子创建按种子名 slug 化后的目录，可选时间戳子目录。
- 使用信号量限制同时处理的种子数量，避免过度并发。

#### 使用方法

```bash
python search_by_images.py \
--input-dir crawledimg/search_by_images/seed \
--output crawledimg/search_by_images/baidu \
--limit-per-seed 150 \
--per-host 6 \
--batch-size 16 \
--max-concurrent 3 \
--dhash-threshold 18 \
--dhash-min-distance 4 \
--timestamp-subdir
```

常用参数：

- --glob：只选择匹配的种子文件。
- --no-phash：关闭 dHash 相似度筛选。
- --no-dedup：关闭跨运行去重。
- --debug：输出调试日志并放宽过滤。

———

### filter_vlm.py —— 多模态模型精筛

#### 实现思路

- 支持两种输入：JSON/JSONL 清单或目录扫描，统一为 Candidate 列表。
- 通过线程池并发请求 VLM。OpenAI 后端使用 Responses API，按 System/User 提示词强制模型返回合法 JSON；Ollama 后端支持本地/云端，使用 requests 调用 /api/chat。
- 对返回 JSON 进行规范化：强制 decision ∈ {keep, reject}，分数截断到 [0,1]，原因截短 160 字符，标签去重小写化。
- 按 min_score 阈值将合格图片复制到 keep_dir，其余复制到 reject_dir（可选）。
- 结果写入 JSONL，同时输出 *.errors.json 汇总失败项。

#### 使用方法

```bash
python filter_vlm.py \
--input-dir filteredimg/baidu/clip_candidates \
--output results/vlm_filter.jsonl \
--backend openai \
--model gpt-4.1 \
--min-score 0.6 \
--workers 4 \
--keep-dir filteredimg/baidu/vlm_keep \
--reject-dir filteredimg/baidu/vlm_reject
```

主要参数：

- --input-manifest：替代 --input-dir。
- --backend：openai / ollama-local / ollama-cloud。
- --model：模型名称，例如 gpt-4.1、llava:latest。
- --timeout：单次请求超时（秒）。
- --limit：仅处理前 N 张，用于抽检。
- --seed：为重试策略设置随机种子。

提示词位于 SYSTEM_PROMPT 与 USER_PROMPT_TEMPLATE。可在 prompt/filter_vlm_v*.txt 中调整后同步更新脚本。

———

## 其他脚本

### filter_basic.py

- clip 模型 (ViT-B/32) 计算图片与室内/室外/tidy/messy/moderate 提示句的相似度。
- 条件判断默认只启用室内判别，可通过 --object-count-min / --object-count-max 激活 Faster R-CNN 物体数量约束。
- 输出到目标目录并记录 clip_filter_log.csv（含各提示词得分、可选物体数）。

使用示例：

```bash
python filter_basic.py \
--input crawledimg/baidu/raw \
--output filteredimg/baidu/clip_keep \
--indoor-margin 0.02 \
--moderate-margin 0.0 \
--object-count-min 8 \
--detector-threshold 0.5 \
--limit 500
```

### util/index_existing.py

- 重建 _global_index.jsonl 或 _filtered_index.jsonl，自动推断模式。
- --dry-run 输出变更统计而不写入。

示例：

```bash
python util/index_existing.py \
--root crawledimg/baidu/raw \
--dry-run
```

### util/remove_seed_duplicates.py

- 计算种子与下载图片的 dHash 距离，删除距种子过近的图片及元数据。
- 默认阈值 2，支持 --dry-run。

示例：

```bash
python util/remove_seed_duplicates.py \
--seed-dir crawledimg/search_by_images/seed \
--output-dir crawledimg/search_by_images/baidu \
--threshold 4
```

———

## 提示词与文档

- prompt/filter_vlm_v*.txt：不同版本 VLM 过滤提示，便于比较与回滚。
- docs/summary.md：当前流程总结、问题分析与优化建议，可作为调参或扩展数据源的参考。

———

## 流程建议与调参提示

1. 先做去重再筛选：util/index_existing.py 可保持 _global_index.jsonl 与磁盘一致，避免重复下载。
2. 逐步收紧筛选：先用 filter_basic.py 快速剔除明显不合格的图片，再用 filter_vlm.py 精选，提高成本效率。
3. 关注日志：crawler/search_by_images 控制台输出保存数量；filter_vlm JSONL/错误文件能定位异常。
4. prompt 调整流程：在 prompt/ 修改提示词后，同步更新脚本中常量或改为读取外部文件，以便实验不同策略。

———

## 许可证与贡献

目前仓库未声明许可证。如需扩展（例如支持更多数据源、引入向量检索排重），欢迎在确保数据合规的前提下提交建议或合并请求。