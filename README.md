# 各ファイルの使い方など

- `run_onefile.sh`: 見ての通り, stderrはそのまま出力され, ファイルは `./tools/out` 以下に保存 `xclip -selection c` がついている
- `run_onefile_noclip.sh`: 並列実行に使う用のやつ
- `exec_testcase.py`: 引数で与えたケース分実行してくれる, `./out_txt/scores.txt` があると相対スコアが出てくる
- `save_scores.py`: `./tools/out` 以下のファイルを読んでスコアをファイルに出力, 引数で最大ファイル数を指定できる
- `classification.py`: 得点ファイルをいくつか渡して分類器を生やす, 複数モードがあって片方コメントアウトしてある
	- タイプ1: 得点最大のものにラベル付けしてDecisionTreeClassifier
	- タイプ2: 相対スコアを得点にしてそれを最大化するように決定木を構築, 分割は深さ先読みせずに全分割を試す貪欲
