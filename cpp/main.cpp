#include <bits/stdc++.h>
// #include "atcoder/all"

// #define FROMFILE

#pragma GCC target("avx2")
#pragma GCC optimize("O2")
#pragma GCC optimize("unroll-loops")

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

template <typename T>
bool chmax(T& x, T y){
    if(x < y){
        x = y;
        return true;
    }
    return false;
}


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



int n, k;
array<int, 11> a;
vector<int> x, y;


// doc: https://shibh308.github.io/library/library/lib/classes/bitvector.cpp.html
struct BitVector{
    vector<uint64_t> v;
    vector<int> r;
    BitVector(){}
    void build(){
        r.assign(v.size() + 1, 0);
        for(int i = 0; i < v.size(); ++i)
            r[i + 1] = r[i] + __builtin_popcountll(v[i]);
    }
    bool access(int x){
        return (v[x >> 6] >> (x & 63)) & 1;
    }
    // [0, x)の1の出現回数
    int rank(int x){
        return r[x >> 6] + __builtin_popcountll(v[x >> 6] & ((1uLL << (x & 63)) - 1));
    }
    int rank(int x, bool fl){
        return fl ? rank(x) : x - rank(x);
    }
};



// doc: https://shibh308.github.io/library/library/lib/classes/waveletmatrix.cpp.html
template <typename T, int W>
struct WaveletMatrix{

    array<BitVector, W> bv;
    array<int, W> zero_cnt;

    WaveletMatrix(vector<T>& a){
        int n = a.size();
        vector<T> v(a);
        for(int i = W - 1; i >= 0; --i){
            vector<uint64_t> b((n >> 6) + 1, 0);
            vector<T> v1, v2;
            for(int j = 0; j < n; ++j){
                ((v[j] >> i) & 1 ? v2 : v1).push_back(v[j]);
                b[j >> 6] |= uint64_t((v[j] >> i) & 1) << (j & 63);
            }
            for(int j = 0; j < v.size(); ++j)
                v[j] = (j < v1.size() ? v1[j] : v2[j - v1.size()]);
            bv[i].v = move(b);
            bv[i].build();
            zero_cnt[i] = bv[i].rank(n, 0);
        }
    }

    // [l, r)内のxの数
    int count(int l, int r, T x){
        for(int i = W - 1; i >= 0; --i){
            bool fl = (x >> i) & 1;
            int st = bv[i].rank(l, fl);
            int en = bv[i].rank(r, fl);
            l = (fl ? zero_cnt[i] : 0) + st;
            r = (fl ? zero_cnt[i] : 0) + en;
        }
        return r - l;
    }

    // [l, r)内で[0, x)を満たす値の数
    int count_lower(int l, int r, T x){
        int cnt = 0;
        for(int i = W - 1; i >= 0; --i){
            bool fl = (x >> i) & 1;
            int st = bv[i].rank(l, fl);
            int en = bv[i].rank(r, fl);
            if(fl){
                st += zero_cnt[i];
                en += zero_cnt[i];
                cnt += (bv[i].rank(r, 0) - bv[i].rank(l, 0));
            }
            l = st, r = en;
        }
        return cnt;
    }

    // [l, r)内で[x, y)を満たす値の数
    int count_range(int l, int r, T x, T y){
        return count_lower(l, r, y) - count_lower(l, r, x);
    }

    // 0-indexedでk番目に小さいものを返す
    T kth_min(int l, int r, int k){
        T ans = 0;
        for(int i = W - 1; i >= 0; --i){
            int st = bv[i].rank(l, 0);
            int en = bv[i].rank(r, 0);
            if(en - st <= k){
                k -= en - st;
                l = zero_cnt[i] + bv[i].rank(l, 1);
                r = zero_cnt[i] + bv[i].rank(r, 1);
                ans |= (1uLL << i);
            }
            else{
                l = st, r = en;
            }
        }
        return ans;
    }

    // [l, r)でのx以上最小値
    pair<T, bool> predecessor(int l, int r, T x){
        int idx = count_lower(l, r, x);
        if(idx == r - l){
            return make_pair((1uLL << W) - 1, false);
        }
        return make_pair(kth_min(l, r, idx), true);
    }

    // [l, r)でのx以下最大値
    pair<T, bool> successor(int l, int r, T x){
        int idx = count_lower(l, r, x + 1);
        if(idx == 0)
            return make_pair(0, false);
        return make_pair(kth_min(l, r, idx - 1), true);
    }
};


clock_t st;


void solve(){
    vector<pair<int,int>> v;
    for(int i = 0; i < n; ++i){
        v.emplace_back(x[i], y[i]);
    }
    sort(v.begin(), v.end());
    for(int i = 0; i < n; ++i){
        x[i] = v[i].first;
        y[i] = v[i].second;
    }

    constexpr int v_cut = 27;
    vector<int> v_poses(1, -100000);
    for(int i = 0; i < v_cut - 1; ++i){
        int bo = n * (i + 1) / v_cut;
        int pos = v[bo].first + 1;
        v_poses.emplace_back(pos);
    }
    v_poses.emplace_back(100000);
    vector<pair<pair<int,int>, pair<int,int>>> ans;

    vector<int> x_poses(1, -10000);
    vector<int> y_poses(1, -10000);

    for(int i = 0; i < v_poses.size() - 2; ++i){
        x_poses.emplace_back(v_poses[i + 1]);
    }

    vector<int> v_poses2(1, -100000);
    for(int i = 0; i < v_cut - 1; ++i){
        int bo = n * (i + 1) / v_cut;
        int pos = v[bo].first + 1;
        v_poses2.emplace_back(pos);
    }
    v_poses2.emplace_back(100000);
    for(int i = 0; i < v_poses2.size() - 2; ++i){
        y_poses.emplace_back(v_poses2[i + 1]);
    }
    /*
    vector<map<int,int>> vq(v_cut);
    for(int i = 0; i < v_cut; ++i){
        for(auto p : gr[i]){
            ++vq[i][p.second];
        }
    }
    array<int, 11> cnt{0,0,0,0,0,0,0,0,0,0,0};
    vector<int> bl_cnt(v_cut, 0);
    for(int i = -10000; i < 10000; ++i){
        auto bef = bl_cnt;
        for(int j = 0; j < v_cut; ++j){
            bl_cnt[j] += vq[j][i];
        }
        if(*max_element(bl_cnt.begin(), bl_cnt.end()) > 10){
            bl_cnt = bef;
            for(auto c : bl_cnt){
                ++cnt[c];
            }
            y_poses.emplace_back(i);
            bl_cnt.assign(v_cut, 0);
        }
    }
     */
    x_poses.emplace_back(10000);
    y_poses.emplace_back(10000);

    vector<int> x_st(20007, -1);
    vector<int> x_en(20007, -1);
    vector<int> _y;
    for(int i = n - 1; i >= 0; --i){
        x_st[x[i] + 10000] = i;
    }
    for(int i = 0; i < n; ++i){
        x_en[x[i] + 10000] = i;
        _y.emplace_back(y[i] + 10000);
    }
    for(int i = 0; i < 20006; ++i){
        chmax(x_st[i + 1], x_st[i]);
        chmax(x_en[i + 1], x_en[i]);
    }
    WaveletMatrix<int, 16> wm(_y);

    auto search = [&](int sx, int ex, int sy, int ey){
        int sxp = x_en[sx + 10000] + 1;
        int exp = x_en[ex + 10000 - 1] + 1;
        int syp = sy + 10000 + 1;
        int eyp = ey + 10000;
        chmax(sxp, 0);
        chmin(exp, n);
        chmax(syp, 0);
        chmin(eyp, 20010);
        return wm.count_range(sxp, exp, syp, eyp);
    };

    vector<int> cnt(11, 0);
    for(int i = 0; i < x_poses.size() - 1; ++i){
        for(int j = 0; j < y_poses.size() - 1; ++j){
            int sx = x_poses[i];
            int ex = x_poses[i + 1];
            int sy = y_poses[j];
            int ey = y_poses[j + 1];
            /*
            int ccc = 0;
            for(int k = 0; k < n; ++k){
                if(sx < x[k] && x[k] < ex && sy < y[k] && y[k] < ey){
                    ++ccc;
                }
            }
             */
            int cn = search(sx, ex, sy, ey);
            /*
            if(ccc != cn){
                cout << "YABAI" << endl;
                cout << ccc << " " << cn << endl;
                exit(0);
            }
             */
            if(cn <= 10){
                ++cnt[cn];
            }
        }
    }
    int res = 0;
    for(int i = 0; i < 11; ++i){
        res += min(cnt[i], a[i]);
    }


    double st_temp = 0.5;
    double en_temp = 0.001;
    double temp = st_temp;

    double res_ev = 0.0;
    vector<double> coef(11, 0);
    for(int i = 0; i < 11; ++i){
        coef[i] = 1.0 + 0.05 * i;
        res_ev += coef[i] * min(a[i], cnt[i]);
    }
    int cnc = 0;
    int ma_res = res;
    vector<int> ma_x = x_poses;
    vector<int> ma_y = y_poses;
    while(true){
        // cout << ma_res << ": " << res << endl;
        if((++cnc %= 100) == 0){
            double per = double(clock() - st) / (2.85 * CLOCKS_PER_SEC);
            if(per >= 1.0){
                break;
            }
            temp = st_temp + (en_temp - st_temp) * per;
        }
        // cout << temp << endl;
        if(chmax(ma_res, res)){
            ma_x = x_poses;
            ma_y = y_poses;
        }
        double rn = Rnd::rnd_double();
        if(rn < 0.5){
            int sel = Rnd::rnd() % (x_poses.size() - 2) + 1;
            int l = x_poses[sel - 1] + 1;
            int r = x_poses[sel + 1] - 1;
            if(r - l <= 1){
                continue;
            }
            int p = Rnd::rnd() % (r - l) + l;
            auto diff = cnt;
            for(int i = 0; i < y_poses.size() - 1; ++i){
                int cn;
                cn = search(x_poses[sel - 1], x_poses[sel], y_poses[i], y_poses[i + 1]);
                if(cn <= 10){
                    --diff[cn];
                }
                cn = search(x_poses[sel], x_poses[sel + 1], y_poses[i], y_poses[i + 1]);
                if(cn <= 10){
                    --diff[cn];
                }
                cn = search(x_poses[sel - 1], p, y_poses[i], y_poses[i + 1]);
                if(cn <= 10){
                    ++diff[cn];
                }
                cn = search(p, x_poses[sel + 1], y_poses[i], y_poses[i + 1]);
                if(cn <= 10){
                    ++diff[cn];
                }
            }
            int aft = 0;
            double aft_ev = 0.0;
            for(int j = 0; j < 11; ++j){
                aft += min(a[j], diff[j]);
                aft_ev += min(a[j], diff[j]) * coef[j];
            }
            if(exp((aft_ev - res_ev) / temp) > Rnd::rnd_double()){
                // if(aft >= res){
                res_ev = aft_ev;
                res = aft;
                cnt = diff;
                x_poses[sel] = p;
            }
        }
        else if (rn < 1.0){
            int sel = (Rnd::rnd() % y_poses.size() - 2) + 1;
            int l = y_poses[sel - 1] + 1;
            int r = y_poses[sel + 1] - 1;
            if(r - l <= 1){
                continue;
            }
            int p = Rnd::rnd() % (r - l) + l;
            auto diff = cnt;
            for(int i = 0; i < x_poses.size() - 1; ++i){
                int cn;
                cn = search(x_poses[i], x_poses[i + 1], y_poses[sel - 1], y_poses[sel]);
                if(cn <= 10){
                    --diff[cn];
                }
                cn = search(x_poses[i], x_poses[i + 1], y_poses[sel], y_poses[sel + 1]);
                if(cn <= 10){
                    --diff[cn];
                }
                cn = search(x_poses[i], x_poses[i + 1], y_poses[sel - 1], p);
                if(cn <= 10){
                    ++diff[cn];
                }
                cn = search(x_poses[i], x_poses[i + 1], p, y_poses[sel + 1]);
                if(cn <= 10){
                    ++diff[cn];
                }
            }
            int aft = 0;
            double aft_ev = 0.0;
            for(int j = 0; j < 11; ++j){
                aft += min(a[j], diff[j]);
                aft_ev += min(a[j], diff[j]) * coef[j];
            }
            if(exp((aft_ev - res_ev) / temp) > Rnd::rnd_double()){
                // if(aft >= res){
                res_ev = aft_ev;
                res = aft;
                cnt = diff;
                y_poses[sel] = p;
            }
        }
        // TODO:
        /*
        else{
            int sel = (rnd() % y_poses.size() - 3);
            int l = y_poses[sel] + 1;
            int r = y_poses[sel + 2] - 1;
            if(r - l <= 1){
                continue;
            }
            int p = rnd() % (r - l) + l;
            auto diff = cnt;
            for(int i = 0; i < x_poses.size() - 1; ++i){
                int cn;
                for(int j = 0; j < 3; ++j){
                    cn = search(x_poses[i], x_poses[i + 1], y_poses[sel + j], y_poses[sel + j + 1]);
                    if(cn <= 10){
                        --diff[cn];
                    }
                }
                cn = search(x_poses[i], x_poses[i + 1], y_poses[sel], p);
                if(cn <= 10){
                    ++diff[cn];
                }
                cn = search(x_poses[i], x_poses[i + 1], p, y_poses[sel + 3]);
                if(cn <= 10){
                    ++diff[cn];
                }
            }
            int aft = 0;
            for(int j = 0; j < 11; ++j){
                aft += min(a[j], diff[j]);
            }
            if(aft >= res){
                cout << "ADAPT" << endl;
                exit(0);
                res = aft;
                cnt = diff;
                y_poses.erase(next(y_poses.begin(), sel));
            }
        }
         */
    }

    x_poses = ma_x;
    y_poses = ma_y;

    for(int i = 0; i < x_poses.size() - 2; ++i){
        ans.emplace_back(make_pair(x_poses[i + 1], -10000), make_pair(x_poses[i + 1], 10000));
    }
    for(int i = 0; i < y_poses.size() - 2; ++i){
        ans.emplace_back(make_pair(-10000, y_poses[i + 1]), make_pair(10000, y_poses[i + 1]));
    }

    cout << ans.size() << endl;
    for(auto [p1, p2] : ans){
        cout << p1.first << " " << p1.second << " " << p2.first << " " << p2.second << endl;
    }
#ifdef HAND
    cout << 1.0 * res / accumulate(a.begin(), a.end(), 0.0) << endl;
#endif
}


namespace params{
void load_params(){
    ifstream ifs("../params.txt");
    assert(ifs.is_open());

    // TODO: load params
}
}

void read_file(istream& ifs){
    ifs >> n >> k;
    x.resize(n);
    y.resize(n);
    for(int i = 0; i < 10; ++i){
        ifs >> a[i + 1];
    }
    for(int i = 0; i < n; ++i){
        ifs >> x[i] >> y[i];
    }
}

signed main(){

    st = clock();

#ifdef OPTIMIZE
    params::load_params();
#endif

#ifndef FROMFILE
    // TODO: input
    read_file(cin);
#else
    ifstream ifs("../tools/in/0003.txt");
    assert(ifs.is_open());
    read_file(ifs);
#endif
    a[0] = 0;
    solve();

}
