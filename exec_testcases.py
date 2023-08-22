import os
import multiprocessing
import subprocess
import sys
import seaborn as sns
import matplotlib.pyplot as plt


DEFAULT_NUM_CASES = 50
NUM_PRETEST_CASES = 50

def execute_case(seed):
    # timeoutをまともな値に設定していないので注意！
    result_str = subprocess.run(['sh', 'run_onefile_noclip.sh', f'{seed:04}'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=100.0).stderr
    for l in result_str.split('\n'):
        prefix = 'Score = '
        if l.startswith(prefix):
            # print(float(result_str[len(prefix):]) * NUM_PRETEST_CASES)
            return seed, float(l[len(prefix):])
    print(f'err: {seed}')
    return seed, 0

# kwargsがなくても動く
def execute(cases_st, cases_en):
    num_cases = cases_en - cases_st
    num_threads = max(1, multiprocessing.cpu_count() - 2)
    count = 0
    scores = []
    with multiprocessing.Pool(num_threads) as pool:
        for seed, score in pool.imap_unordered(execute_case, range(cases_st, cases_en)):
            scores.append((score, seed))
            print(count % 10, end='', flush=True)
            count += 1
    print()
    print()
    scores.sort()
    scores.reverse()

    ratio_sum_base = 0.0
    ratio_sum = 0.0
    now_score = 22.893

    improve = 0
    worse = 0
    tooimprove = 0
    tooworse = 0
    ratios = [0 for _ in range(len(scores))]
    with open('./out_txts/scores.txt') as f:
        base_scores = list(map(int, f.readlines()))
        for (score, seed) in scores:
            base = base_scores[seed] if seed < len(base_scores) else 0
            ratio = score / base
            print(f'{seed:04}: {ratio:0>.3} {base:10} => {score:10}')
            improve += ratio > 1.0
            worse += ratio < 1.0
            ratios[seed - cases_st] = ratio
            if ratio > 10.0:
                tooimprove += 1
            if ratio < 0.1:
                tooworse += 1

            ratio_sum += score / max(score, base) * 1.5
            ratio_sum_base += base / max(score, base) * 1.5

    # sns.histplot(ratios, bins=100, log_scale=True)
    # plt.savefig('ratio_bin.png')

    # print('worst:', scores[0]) # [:10])
    # print('best:', scores[-1]) # scores[-10::-1])
    print('improve:', improve)
    print('worse  :', worse)
    print('improve 10:', tooimprove)
    print('worse   10:', tooworse)
    total = sum([s[0] for s in scores])
    ave = total / num_cases
    result = {}
    result['cases'] = num_cases
    result['ratio'] = ratio_sum / ratio_sum_base
    # result['total'] = total
    result['min'] = (scores[0][1], scores[0][0])
    result['max'] = (scores[-1][1], scores[-1][0])
    result['ave'] = ave
    result['submit/10^6'] = (ave * NUM_PRETEST_CASES) / 10**6
    result['relative'] = (ratio_sum / num_cases) * now_score
    return result


if __name__ == '__main__':
    subprocess.run(['cargo', 'build', '--manifest-path=./rs/Cargo.toml', '--release'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cases_st = 0
    cases_en = DEFAULT_NUM_CASES
    if len(sys.argv) == 2:
        cases_en = int(sys.argv[1])
    elif len(sys.argv) == 3:
        cases_st = int(sys.argv[1])
        cases_en = int(sys.argv[2])
    else:
        abort()

    result = execute(cases_st, cases_en)
    for key, value in result.items():
        if key != 'ratio' and key != 'submit/10^6':
            print(f'{key:12}: {value}')
    for key, value in result.items():
        if key == 'submit/10^6':
            print(f'{key:12}: {value}')
    for key, value in result.items():
        if key == 'ratio':
            print(f'{key:12}: {value}')