# Changelog

## v0.5.0 - 2026-08-14

- **LLM 工具后台分析**：`analyze_current_audio` / `analyze_audio_by_number` 不再同步等待结果，改为提交后台任务立即返回，分析完成后唤醒主 Agent 以角色口吻主动发送结果，规避 AstrBot 120 秒工具调用超时（对齐 video_sense 机制）。
- **音频感知提示注入**：收到新音频后，在 LLM 请求上下文中注入一次性「[音频感知]」提示（每音频仅一次，不消耗 API），引导 LLM 主动调用分析工具。
- **列表显示相对时间**：`list_audio_files`、指令无序号提示、`analyze_current_audio` 无附件时的缓存列表均标注「刚刚/N分钟前/N小时前/N天前」，便于 LLM 分辨新旧音频。
- **缓存去重与上限**：同名同引用的音频只缓存一次（重复引用仅刷新接收时间）；新增配置「缓存上限（条）」默认 50，超出自动裁剪最旧条目，防止长会话内存膨胀。
- **追问参数**：`analyze_current_audio` 新增可选 `question` 参数，LLM 可将用户追问传入分析提示词。
- **网关 HTML 错误页识别**：API 返回 HTML（如 nginx 502 页面）时给出友好提示与排查建议（降低文件大小 / 更换接入方）。
- 新增 pyproject.toml（ruff 配置，与 video_sense 一致）；版本号 v0.4.0 → v0.5.0。
