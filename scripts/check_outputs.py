#!/usr/bin/env python3
"""驗證 docs/*.html 裡「指令 → 輸出」的每一對都還成立。

教材裡每個 <pre class="shell"> 後面緊跟的 <pre class="out">，就是一組宣稱：
「照這條指令跑，會印出這些東西」。程式碼會繼續演進，這些宣稱會靜靜過期。
這支腳本把指令實際跑起來，跟教材寫的輸出逐行比對。

用法（在 repo 根目錄執行，且已 source .venv/bin/activate）:
    python scripts/check_outputs.py [docs_dir]

結束碼: 0 = 全部相符、1 = 有對不上的、2 = 環境或參數有問題。
"""
import html
import os
import re
import subprocess
import sys
from pathlib import Path

if len(sys.argv) > 2 or any(a.startswith("-") for a in sys.argv[1:]):
    print(f"用法: {sys.argv[0]} [docs_dir]", file=sys.stderr)
    sys.exit(2)

DOCS = Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
if not DOCS.is_dir():
    print(f"找不到目錄: {DOCS}", file=sys.stderr)
    sys.exit(2)
if not Path("chase_2d/chase_2d.py").exists():
    print("請在 repo 根目錄執行", file=sys.stderr)
    sys.exit(2)

# 教材裡刻意改寫過的部分：比對時兩邊都套用同樣的正規化
NORMALIZE = [
    (re.compile(r"^(Q 表已存成|學習曲線已存成) .*/(rl_playground/.*)$"), r"\1 <path>/\2"),
    (re.compile(r"^&lt;你的路徑&gt;.*$"), ""),
]
MARK = re.compile(r"\s*<span class=\"mark\">.*?</span>\s*$")


def clean(block, strip_marks):
    out = []
    for line in block.split("\n"):
        if strip_marks:
            line = MARK.sub("", line)
        line = html.unescape(re.sub(r"<[^>]+>", "", line)).rstrip()
        line = re.sub(r"^(Q 表已存成|學習曲線已存成) .*?(/rl_playground/.*)$", r"\1 <path>\2", line)
        line = re.sub(r"^(Q 表已存成|學習曲線已存成) <你的路徑>(/rl_playground/.*)$", r"\1 <path>\2", line)
        out.append(line)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


BLOCK = re.compile(r'<pre class="(shell|out)">(.*?)</pre>', re.S)


def pairs_in(text):
    """每個 shell 區塊配上「下一個 out 區塊」，中間不得再有 shell 區塊。

    配不到對的 shell 區塊會以 None 回報，而不是被靜靜略過 ——
    悄悄跳過的區塊看起來跟通過一模一樣。
    """
    blocks = [(m.group(1), m.group(2)) for m in BLOCK.finditer(text)]
    result = []
    for i, (kind, body) in enumerate(blocks):
        if kind != "shell":
            continue
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        result.append((body, nxt[1] if nxt and nxt[0] == "out" else None))
    return result


env = dict(os.environ, MPLBACKEND="Agg")
failed = False
checked = skipped = 0

for page in sorted(DOCS.glob("*.html")):
    pairs = pairs_in(page.read_text(encoding="utf-8"))
    if not pairs:
        continue
    print(f"\n== {page.name}：{len(pairs)} 組指令／輸出 ==")
    for i, (cmd_raw, out_raw) in enumerate(pairs, 1):
        cmd = html.unescape(re.sub(r"<[^>]+>", "", cmd_raw)).strip()
        if "git clone" in cmd:
            print(f"  {i}. 跳過（clone / venv 建置）")
            skipped += 1
            continue
        if out_raw is None:
            failed = True
            print(f"  {i}. 這個指令區塊後面沒有緊接輸出區塊，無法驗證: {cmd.splitlines()[0][:50]}")
            continue
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env)
        actual = [l for l in clean(proc.stdout, False)
                  if "FigureCanvasAgg" not in l and not l.strip().startswith("plt.")]
        expected = clean(out_raw, True)
        checked += 1
        if actual == expected:
            print(f"  {i}. OK（{len(expected)} 行）")
        else:
            failed = True
            print(f"  {i}. 不符：")
            for n, (e, a) in enumerate(zip(expected, actual)):
                if e != a:
                    print(f"       行 {n+1} 教材: {e!r}")
                    print(f"       行 {n+1} 實際: {a!r}")
            if len(expected) != len(actual):
                print(f"       行數 教材 {len(expected)} vs 實際 {len(actual)}")

print(f"\n驗證 {checked} 組，跳過 {skipped} 組：", "有對不上的，見上方" if failed else "全部相符")
sys.exit(1 if failed else 0)
