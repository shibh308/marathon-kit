#include <bits/stdc++.h>
// #include "atcoder/all"

// #define FROMFILE

#pragma GCC target("avx2")
#pragma GCC optimize("Ofast")
// #pragma GCC optimize("unroll-loops")

using namespace std;

using i64 = long long;

const i64 MOD = 1e9 + 7;
const i64 INF = i64(1e18);

template <typename T>
bool chmin(T& x, T y){
    if(x > y){
        x = y;
        return true;
    }
    return false;
}



// doc: https://shibh308.github.io/library/library/lib/functions/xorshift.cpp.html
namespace Rnd{
// doc: https://shibh308.github.io/library/library/lib/functions/xorshift.cpp.html
uint64_t x = 0xdeadbeef0110dead;
uint64_t rnd(){
    x ^= x << 7;
    x ^= x >> 9;
    return x;
}
uint64_t rnd(int n){
    return rnd() % n;
}
double rnd_double(){
    return 1.0 * rnd() / numeric_limits<uint64_t>::max();
}
vector<int> rnd_perm(int n){
    vector<int> v(n);
    iota(v.begin(), v.end(), 0);
    for(int i = n - 1; i >= 1; --i){
        int j = rnd(i + 1);
        swap(v[i], v[j]);
    }
    return v;
}
template<typename T>
void shuffle(vector<T>& v){
    int n = v.size();
    for(int i = n - 1; i >= 1; --i){
        int j = rnd(i + 1);
        swap(v[i], v[j]);
    }
}
}

using namespace Rnd;


template <typename T>
bool chmax(T& x, T y){
    if(x < y){
        x = y;
        return true;
    }
    return false;
}


namespace params{
void load_params(){
    ifstream ifs("../params.txt");
    assert(ifs.is_open());

    // TODO: load params
}
}

void read_file(istream& ifs){
    // TODO: read from file
}



constexpr int d = 365;
constexpr int n = 26;
void solve(){
    int d_;
    cin >> d_;
    vector<int> pena(n);
    for(int i = 0; i < n; ++i){
        cin >> pena[i];
    }
    vector<vector<int>> satis(n);
    vector<double> ave(n), dist(n);

    vector<int> plan(d);
    for(int i = 0; i < d; ++i){
        plan[i] = i % n;
    }

    auto replan = [&](int day, vector<double>& ave){

        double score = 0;
        vector<vector<int>> plan_inv(n);
        for(int i = 0; i < d; ++i){
            plan_inv[plan[i]].emplace_back(i);
        }

        auto pen = [&](int typ, int between){
            assert(between >= 1);
            return (1LL * between * (between - 1) / 2) * pena[typ];
        };

        auto change = [&](int x, int aft){
            int bef = plan[x];
            plan[x] = aft;

            auto iter_bef = lower_bound(plan_inv[bef].begin(), plan_inv[bef].end(), x);
            int bef_l = iter_bef == plan_inv[bef].begin() ? -1 : *prev(iter_bef);
            int bef_r = next(iter_bef) == plan_inv[bef].end() ? d : *next(iter_bef);

            auto iter_aft = lower_bound(plan_inv[aft].begin(), plan_inv[aft].end(), x);
            int aft_l = iter_aft == plan_inv[aft].begin() ? -1 : *prev(iter_aft);
            int aft_r = iter_aft == plan_inv[aft].end() ? d : *iter_aft;

            score -= ave[bef];
            score += pen(bef, x - bef_l);
            score += pen(bef, bef_r - x);
            score -= pen(bef, bef_r - bef_l);

            score += ave[aft];
            score -= pen(aft, x - aft_l);
            score -= pen(aft, aft_r - x);
            score += pen(aft, aft_r - aft_l);

            plan_inv[bef].erase(iter_bef);
            plan_inv[aft].insert(iter_aft, x);
        };

        auto check = [&](int iter, double bef_score, double aft_score){
            return bef_score <= score;
        };

        int remain = d - day;
        for(int iter = 0; iter < 100000; ++iter){
            if(rnd(2)){
                int x = rnd(remain) + day;
                int to;
                do{
                    to = rnd(n);
                }while(plan[x] == to);
                int bef_plan = plan[x];

                double bef_score = score;
                change(x, to);
                if(check(iter, bef_score, score)){
                }
                else{
                    // cout << bef_score << " -> " << score << " -> ";
                    change(x, bef_plan);
                    // cout << score << endl;
                    assert(abs(bef_score - score) <= 1e-7);
                }
            }
            else{
                int l, r;
                int cn = 0;
                if(remain >= 15){
                    l = rnd(remain - 10) + day;
                    do{
                        r = l + 1 + rnd(9);
                    }while(plan[l] == plan[r] && ++cn < 10);
                }
                else{
                    l = rnd(remain) + day;
                    do{
                        r = rnd(remain) + day;
                    }while(plan[l] == plan[r] && ++cn < 10);
                }
                if(cn >= 10){
                    continue;
                }

                int bef_l = plan[l];
                int bef_r = plan[r];

                double bef_score = score;
                change(l, bef_r);
                change(r, bef_l);
                if(check(iter, bef_score, score)){
                }
                else{
                    change(l, bef_l);
                    change(r, bef_r);
                    assert(abs(bef_score - score) <= 1e-7);
                }
            }
        }
        // cout << day << " " << score << endl;
    };

    vector<int> rem(n, 0);
    i64 score = 0;
    for(int day = 1; day <= d; ++day){
        for(int i = 0; i < n; ++i){
            int x;
            cin >> x;
            satis[i].emplace_back(x);

            ave[i] = 0.0;
            for(int j = 0; j < day; ++j){
                ave[i] += satis[i][j];
            }
            ave[i] /= (day + 1);
            double dist_sq = 0.0;
            for(int j = 0; j < day; ++j){
                dist_sq += pow(satis[i][j] - ave[i], 2.0);
            }
            dist[i] = sqrt(dist_sq);
        }

        replan(day - 1, ave);

        int sel = plan[day - 1];
        // int sel = day % n;

        score += satis[sel][day - 1];
        for(int i = 0; i < n; ++i){
            if(i != sel){
                ++rem[i];
                score -= pena[i] * rem[i];
            }
            else{
                rem[i] = 0;
            }
        }
        cout << sel + 1 << endl;
        // cerr << score << endl;
    }
    cerr << score << endl;
}


signed main(){
    clock_t st = clock();

#ifdef OPTIMIZE
    params::load_params();
#endif

#ifdef NOSUBMIT
#endif

    solve();

    /*
#ifndef FROMFILE
    // TODO: input
    read_file(cin);
#else
    ifstream ifs("../tools/in/0003.txt");
    assert(ifs.is_open());
    read_file(ifs);
#endif
     */

}
