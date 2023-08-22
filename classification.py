import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def parse_tree(data_x, data_y, num_classes, clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    parents = np.zeros(shape=n_nodes, dtype=np.int64)
    isright = np.zeros(shape=n_nodes, dtype=bool)
    parents[0] = -1
    stack = [(0, 0)]

    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            parents[children_left[node_id]] = node_id
            parents[children_right[node_id]] = node_id
            isright[children_left[node_id]] = False
            isright[children_right[node_id]] = True
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    rules = []
    for i in range(n_nodes):
        if is_leaves[i]:
            l = []
            idx = i
            elms = np.ones(shape=data_y.shape, dtype=bool)
            while idx != 0:
                right = isright[idx]
                idx = parents[idx]
                f  = feature[idx]
                th = int(threshold[idx])
                mask = (data_x[:, f] > th) if right else (data_x[:, f] <= th)
                l.append((f, right, th))
                elms = np.bitwise_and(elms, mask)
            cnts = []
            for c in range(len(data_x[0])):
                cnts.append((data_y[elms] == c).sum())
            y = cnts.index(max(cnts))
            rules.append((l, y))
    print('Classifier(vec![')
    for rule, y in rules:
        print('(vec![', end='')
        for (f, right, th) in rule:
            r = 'true' if right else 'false'
            print(f'({f}, {r}, {th})', end=',')
        print(f'], {y}),')
    print('])')

def get_score_mat():
    num_classes = len(sys.argv) - 1
    num_cases = 10000000000
    # 転置してないので注意！
    score_mat = []
    for i in range(num_classes):
        with open(sys.argv[i + 1]) as f:
            scores = list(map(int, f.readlines()))
            num_cases = min(num_cases, len(scores))
            score_mat.append(scores)
    return score_mat, num_cases

def get_score_and_features():
    score_mat, num_cases = get_score_mat()

    data_x = []
    for i in range(num_cases):
        with open(f'./tools/in/{i:04}.txt') as f:
            x = list(map(int, f.readline().split()))
            data_x.append(x)
            with open(f'./tools/out/{i:04}.txt') as f:
                k = int(f.readline().split()[1])
                data_x[-1].append(k)
                k = int(f.readline().split()[1])
                data_x[-1].append(k)
                k = int(f.readline().split()[1])
                data_x[-1].append(k)
    return data_x, score_mat, num_cases


def get_score_data():
    num_classes = len(sys.argv) - 1
    data_x, score_mat, num_cases = get_score_and_features()

    data_y = []
    for i in range(num_cases):
        l = []
        for j in range(num_classes):
            l.append(score_mat[j][i])
        data_y.append(l.index(max(l)))
    return num_classes, data_x, data_y


def get_high_mat():
    i = 0
    data_x = []
    data_y = []
    while True:
        if not os.path.exists(f'./tools/out/{i:04}.txt'):
            break
        with open(f'./tools/in/{i:04}.txt') as f:
            x = list(map(int, f.readline().split()))
            data_x.append(x)
        with open(f'./tools/out/{i:04}.txt') as f:
            k = int(f.readline().split()[1])
            if k < 6:
                data_y.append(k - 1)
            elif k < 11:
                data_y.append(5)
            else:
                data_y.append(6)
        i += 1
    return 7, data_x, data_y


# 予測率を最大化するclassification
def main_classification():
    # ファイルをいくつか受け取り, スコア最大のものを取る
    num_classes, data_x, data_y = get_score_data()
    
    # in/outから読み取り, num_highの予測を出す
    # 1, 2, 3, 4, 5, 678910, 11... という分類なので注意
    # num_classes, data_x, data_y = get_high_mat()

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    tree = DecisionTreeClassifier(max_leaf_nodes=10)

    tree.fit(data_x, data_y)
    parse_tree(data_x, data_y, num_classes, tree)


# ルールリストの焼きなまし
# 相対スコアなどの謎評価指標を使いたいときとかに使う
def main_greedy_tree():
    num_features = 3
    num_classes = len(sys.argv) - 1
    data_x, score_mat, num_cases = get_score_and_features()

    data_x = np.array(data_x)

    spls = [
        list(range(10, 50)),
        list(range(60, 100)),
        [x ** 2 for x in range(1, 30)],
        list(range(1,200)),
        list(range(0, 1000)),
        list(range(0, 10))
    ]

    data_y = []
    for i in range(num_cases):
        scores = [score_mat[j][i] for j in range(num_classes)]
        max_score = max(scores) * 1.5
        scores = list(map(lambda x: x / max_score, scores))
        data_y.append(scores)

    data_y = np.array(data_y)

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, )

    leaves = [[]]
    for dep in range(4):
        # 1つの特徴量で貪欲に分割していく
        nex_leaves = []
        for leaf in leaves:
            mask = np.ones(len(train_x), bool)
            for (f, op, val) in leaf:
                if op:
                    mask = mask & (train_x[:, f] > val)
                else:
                    mask = mask & ~(train_x[:, f] > val)
            opt = (-100, None, None)
            for f in range(num_features):
                for val in spls[f]:
                    mask_1 = mask & (train_x[:, f] > val)
                    mask_0 = mask & ~(train_x[:, f] > val)
                    score_1 = 0
                    score_0 = 0
                    for ans in range(num_classes):
                        score_1 = max(score_1, train_y[mask_1, ans].sum())
                        score_0 = max(score_0, train_y[mask_0, ans].sum())
                    score = score_1 + score_0
                    if opt[0] < score:
                        opt = (score, f, val)
            nex_leaves.append(leaf + [(opt[1], False, opt[2])])
            nex_leaves.append(leaf + [(opt[1], True, opt[2])])
        leaves = nex_leaves

    train_score_sum = 0.0
    test_score_sum = 0.0
    print('Classifier(vec![')
    for leaf in leaves:
        mask = np.ones(len(train_x), bool)
        for (f, op, val) in leaf:
            if op:
                mask = mask & (train_x[:, f] > val)
            else:
                mask = mask & ~(train_x[:, f] > val)
        train_scores = []
        for ans in range(num_classes):
            train_scores.append(train_y[mask, ans].sum())

        mask = np.ones(len(test_x), bool)
        for (f, op, val) in leaf:
            if op:
                mask = mask & (test_x[:, f] > val)
            else:
                mask = mask & ~(test_x[:, f] > val)
        test_scores = []
        for ans in range(num_classes):
            test_scores.append(test_y[mask, ans].sum())

        max_sc = max(train_scores)
        ans = train_scores.index(max_sc)
        train_score_sum += train_scores[ans]
        test_score_sum += test_scores[ans]
        # print(leaf, ans)
        # print(leaves)
        print('(vec![', end='')
        for (f, op, val) in leaf:
            r = 'true' if op else 'false'
            print(f'({f}, {r}, {val})', end=',')
        print(f'], {ans}),')
    print('])')

    print()
    print('train')
    for i in range(num_classes):
        print(i, train_y[:, i].sum())
    print(train_score_sum)
    print()
    print('test')
    for i in range(num_classes):
        print(i, test_y[:, i].sum())
    print(test_score_sum)


if __name__ == '__main__':
    # main_classification()
    main_greedy_tree()

