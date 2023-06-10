import os
import multiprocessing
import subprocess


DEFAULT_NUM_CASES = 50
NUM_PRETEST_CASES = 100

def execute_case(seed):
    input_file_path = f'tools/in/{seed:04}.txt'
    # timeoutをまともな値に設定していないので注意！
    result_str = subprocess.run(['sh', 'run_onefile.sh', input_file_path, f'{seed:04}'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=100.0).stdout
    prefix = 'Score = '
    assert result_str.startswith(prefix)
    return seed, float(result_str[len(prefix):])

# kwargsがなくても動く
def execute(num_cases=DEFAULT_NUM_CASES):
    num_threads = multiprocessing.cpu_count() # max(1, multiprocessing.cpu_count() - 1)
    count = 0
    scores = []
    with multiprocessing.Pool(num_threads) as pool:
        for seed, score in pool.imap_unordered(execute_case, range(1, num_cases)):
            scores.append((score, seed))
            print(count % 10, end='', flush=True)
            count += 1
    print()
    print()
    scores.sort()
    total = sum([s[0] for s in scores])
    ave = total / num_cases
    result = {}
    result['cases'] = num_cases
    result['total'] = total
    result['submit'] = ave * NUM_PRETEST_CASES
    result['min'] = (scores[0][1], scores[0][0])
    result['max'] = (scores[-1][1], scores[-1][0])
    result['ave'] = ave
    return result


if __name__ == '__main__':
    result = execute()
    for key, value in result.items():
        print(f'{key:6}: {value}')