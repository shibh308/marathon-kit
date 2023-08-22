import os
import sys
import multiprocessing
import subprocess


DEFAULT_NUM_CASES = 1000
NUM_PRETEST_CASES = 50

def execute(seed):
    in_file = f'./tools/in/{seed:04}.txt'
    out_file = f'./tools/out/{seed:04}.txt'
    # timeoutをまともな値に設定していないので注意！
    result_str = subprocess.run(['cargo', 'run', '--manifest-path=./tools/Cargo.toml', '--release', '--bin', 'vis', in_file, out_file], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=100.0).stderr
    # result_str = subprocess.run(['cargo', 'run', '--manifest-path=./tools/Cargo.toml', '--release', '--bin', 'vis', in_file, out_file], stdout=subprocess.PIPE, text=True, timeout=100.0).stdout
    for l in result_str.split('\n'):
        prefix = 'Score = '
        if l.startswith(prefix):
            return (seed, int(l[len(prefix):]))
    print(f'err: {seed}')
    return 0

def main():
    if len(sys.argv) == 1:
        print('Usage: python3 save_scores.py ./out_txts/scores_file.txt')
    assert len(sys.argv) >= 2
    out_file_path = sys.argv[1]
    case_max = 1000000000
    if len(sys.argv) >= 3:
        case_max = int(sys.argv[2])
    
    num_cases = 0
    while True:
        if not os.path.exists(f'./tools/out/{num_cases:04}.txt'):
            break
        num_cases += 1
        if num_cases == case_max:
            break
    scores = [0 for i in range(num_cases)]

    count = 0
    num_threads = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(num_threads) as pool:
        for seed, score in pool.imap_unordered(execute, range(num_cases)):
            print(count % 10, end='', flush=True)
            count += 1
            scores[seed] = score

    with open(out_file_path, 'w') as f:
        for i in range(num_cases):
            f.write(str(scores[i]) + '\n')

    score_sum = sum(scores)
    print()
    print('ave:', score_sum / num_cases)
    print('submit/10^6:', (score_sum / num_cases * NUM_PRETEST_CASES) / 10**6)


if __name__ == '__main__':
    main()
