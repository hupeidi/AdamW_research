# Tiny Shakespeare

本项目使用 Karpathy 的 char-rnn 仓库所附教学语料。原始作品为 Shakespeare 的文本，当前文件组织来自该公开仓库；引用与来源保留如下。

- 目录：<https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare>
- 原始下载地址：<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>
- 获取日期：2026-10-30（Asia/Shanghai）
- 文件：`input.txt`，1,115,394 bytes / characters，UTF-8
- SHA-256：`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`

使用连续切分 `[0,1003854)`、`[1003854,1059624)`、`[1059624,1115394)` 分别作为 train、val、test。词表从训练段生成。训练程序检查哈希，不静默接受不同语料。
