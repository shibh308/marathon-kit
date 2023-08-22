use ordered_float::NotNan;
use proconio::input;
use proconio::source::line::LineSource;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::env::join_paths;
use std::f64::consts::PI;
use std::io;
use std::io::{stdout, BufReader, Write};
use std::process::exit;
use std::time::Instant;

#[derive(Debug, Clone)]
struct RandXor {
    x: u64,
}

impl RandXor {
    fn new() -> Self {
        RandXor {
            x: 88172645463325252,
        }
    }
    #[inline(always)]
    fn rand(&mut self) -> u64 {
        self.x ^= (self.x << 7);
        self.x ^= (self.x >> 9);
        self.x
    }
    #[inline(always)]
    fn rand_n(&mut self, n: usize) -> usize {
        self.rand() as usize % n
    }
    #[inline(always)]
    fn rand_range(&mut self, l: usize, r: usize) -> usize {
        l + self.rand() as usize % (r - l)
    }
    #[inline(always)]
    fn rand_f64(&mut self) -> f64 {
        let bits = 1.0_f64.to_bits();
        const EXP: usize = 52;
        let exp_bits = self.rand() & ((1 << EXP) - 1);
        let mask = (((1u64 << 12) - 1) << EXP);
        let new_bits = (bits & mask) | exp_bits;
        let f = f64::from_bits(new_bits);
        f - 1.0
    }
    fn shuffle<T: Clone>(&mut self, v: &mut Vec<T>) {
        for i in (1..v.len()).rev() {
            let j = self.rand_n(i + 1);
            let tmp = v[i].clone();
            v[i] = v[j].clone();
            v[j] = tmp;
        }
    }
    fn sample_norm(&mut self) -> f64 {
        let x = self.rand_f64();
        let y = self.rand_f64();
        (-2. * x.ln()).sqrt() * (2. * PI * y).cos()
    }
}

// [[(feature, <=, val),...], pred]
struct Classifier(Vec<(Vec<(usize, bool, usize)>, usize)>);
impl Classifier {
    fn pred(
        &self,
        n: usize,
        m: usize,
        sd: f64,
        num_high: usize,
        high_val: usize,
        radius: usize,
    ) -> usize {
        let features = vec![n, m, sd as usize, num_high, high_val, radius];
        for (rule, pred) in &self.0 {
            if rule
                .iter()
                .all(|&(f, right, val)| (features[f] <= val) ^ right)
            {
                return *pred;
            }
        }
        unreachable!()
    }
}

fn norm_density_func(x: f64) -> f64 {
    1. / ((2. * PI).sqrt()) * (-0.5 * x * x).exp()
}

#[derive(Clone)]
struct Input {
    n: usize,
    m: usize,
    sd: f64,
    poses: Vec<(usize, usize)>,
    answer: Option<Vec<usize>>,
}
impl Input {
    fn new(simulation: bool) -> Self {
        let stdin = io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            n: usize,
            m: usize,
            sd: f64,
            poses: [(usize, usize); m],
        }
        let answer = if simulation {
            input! {
                from &mut source,
                v: [usize; m]
            }
            Some(v)
        } else {
            None
        };
        Self {
            n,
            m,
            sd,
            poses,
            answer,
        }
    }
}

fn make_prob_table(sd: f64) -> Vec<f64> {
    /*
    range_max = 1000 / sd まで区分求積をしている
    区分求積の間隔 duration は (1000 / sd) / (1000 * 1000)
    このとき, [0, σ] では sigma_num = 1.0 / duration = 1000 * 1000 / range_max 回サンプリングしている
    したがって, 分布の値の総和は sigma_num * 0.5 （片側なことを考慮）となる
     */
    let mut bin = vec![0.0; 1001];
    for i in 0..(10000 * 1000) {
        let val = i as f64 / 10000.0;
        let r = val.round() as usize;
        bin[r] += norm_density_func(val / sd);
    }
    let range_max = 1000.0 / sd;
    // println!("range max: {}", 1000. / sd);
    // let duration = 1.0 / (1000.0 * sd);
    // let sigma_num = 1.000 / duration;
    let sigma_num = (10000.0 * 1000.0 / range_max) as usize;
    let sum = sigma_num as f64 * 0.5;
    /*
    println!("sigma sum: {}", sigma_sum);
    println!("sum: {}", sum);
    println!();
     */
    let bin = bin.iter().map(|&x| x / sum).collect::<Vec<f64>>();
    /*
    for i in 0..=1000 {
        println!("{}", bin[i] * bin.len() as f64);
    }
     */
    bin
}

#[derive(Debug, Clone)]
struct Env {
    n: usize,
    m: usize,
    sd: f64,
    turn: usize,
    output: bool,
    poses: Vec<(usize, usize)>,
    logs: Vec<Vec<(i64, i64, usize)>>,
    rng: RandXor,
    prob_table: Vec<f64>,
    prob_table_cum: Vec<f64>,
    answer: Option<Vec<usize>>,
    err: Option<usize>,
    map: Option<Vec<Vec<usize>>>,
    placement_cost: Option<usize>,
    measure_cost: usize,
    num_high: usize,
    high_val: usize,
    second_high: usize,
    lb: Option<usize>,
    radius: usize,
}

impl Env {
    fn get_move(&self, (x, y): (usize, usize), (x2, y2): (usize, usize)) -> (i64, i64) {
        let xd = x2 as i64 - x as i64;
        let yd = y2 as i64 - y as i64;
        let n = self.n as i64;
        let xd_minus = if xd <= 0 { -xd } else { n - xd };
        let xd_plus = if xd >= 0 { xd } else { n + xd };
        let yd_minus = if yd <= 0 { -yd } else { n - yd };
        let yd_plus = if yd >= 0 { yd } else { n + yd };
        assert!(xd_minus >= 0);
        assert!(xd_plus >= 0);
        assert!(yd_minus >= 0);
        assert!(yd_plus >= 0);
        let xd = if xd_plus <= xd_minus {
            xd_plus
        } else {
            -xd_minus
        };
        let yd = if yd_plus <= yd_minus {
            yd_plus
        } else {
            -yd_minus
        };
        (xd, yd)
    }
    fn che_dist(&self, (x, y): (usize, usize), (x2, y2): (usize, usize)) -> usize {
        let xd = (x as i64 - x2 as i64).abs() as usize;
        let xd = xd.min(self.n - xd);
        let yd = (y as i64 - y2 as i64).abs() as usize;
        let yd = yd.min(self.n - yd);
        xd.max(yd)
    }
    fn man_dist(&self, (x, y): (usize, usize), (x2, y2): (usize, usize)) -> usize {
        let xd = (x as i64 - x2 as i64).abs() as usize;
        let xd = xd.min(self.n - xd);
        let yd = (y as i64 - y2 as i64).abs() as usize;
        let yd = yd.min(self.n - yd);
        xd + yd
    }
    fn add_pos(&self, (x, y): (usize, usize), (dx, dy): (i64, i64)) -> (usize, usize) {
        (
            ((self.n + x) as i64 + dx) as usize % self.n,
            ((self.n + y) as i64 + dy) as usize % self.n,
        )
    }
    fn probability_cum(&self, truth: usize, observed: usize) -> f64 {
        if truth == observed {
            1.0
        } else {
            let diff = (truth as i64 - observed as i64).abs() as usize;
            1.0 - self.prob_table_cum[diff]
        }
    }
    fn probability(&self, truth: usize, observed: usize) -> f64 {
        let prob = if observed != 0 && observed != 1000 {
            if truth == observed {
                self.prob_table[0]
            } else {
                self.prob_table[(truth as i64 - observed as i64).abs() as usize] / 2.0
            }
        } else if observed == truth {
            // (0, 0), (1000, 1000)
            0.5 + self.prob_table[0] / 2.0
        } else {
            // (*, 1000)
            let diff = if observed == 0 { truth } else { 1000 - truth };
            0.5 - self.prob_table_cum[diff] / 2.0
        };
        prob.max(0.0)
    }
    fn output_map(&mut self, map: &Vec<Vec<usize>>) {
        self.map = Some(map.clone());
        if !self.output {
            return;
        }
        for i in 0..self.n {
            for j in 0..self.n {
                print!("{}", map[i][j]);
                if j + 1 != self.n {
                    print!(" ");
                } else {
                    println!();
                }
            }
        }
        stdout().flush().unwrap();
    }
    fn output_pred(&self, preds: &Vec<usize>) {
        if !self.output {
            return;
        }
        println!("-1 -1 -1");
        for i in 0..self.m {
            println!("{}", preds[i]);
        }
        stdout().flush().unwrap();
    }
    fn sample(&mut self, map: &Vec<Vec<usize>>, hole: usize, (dx, dy): (i64, i64)) -> usize {
        self.turn += 1;
        self.measure_cost += 100 * (10 + dx.abs() + dy.abs()) as usize;
        let stdin = io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        if self.output {
            println!("{} {} {}", hole, dx, dy);
            stdout().flush().unwrap();
        }
        let value = match &self.answer {
            None => {
                input! {
                    from &mut source,
                    value: usize
                }
                value
            }
            Some(answer) => {
                let j = answer[hole];
                let (x, y) = self.add_pos(self.poses[j], (dx, dy));
                let val = map[x][y];
                let observed = val as f64 + self.rng.sample_norm() * self.sd;
                (observed.round()).min(1000.0).max(0.0) as usize
            }
        };
        self.logs[hole].push((dx, dy, value));
        value
    }
    fn move_list(&self) -> Vec<(i64, i64)> {
        let mut v = vec![];
        for i in 0..self.n {
            for j in 0..self.n {
                let (dx, dy) = self.get_move((0, 0), (i, j));
                v.push((self.man_dist((0, 0), (i, j)), dx, dy));
            }
        }
        v.sort();
        v.into_iter().map(|(x, i, j)| (i, j)).collect()
    }
    fn measure_cost(&self) -> usize {
        self.measure_cost
        /*
        let mut sum = 0;
        for v in &self.logs {
            for &(x, y, _) in v {
                sum += 100 * (10 + x.abs() + y.abs()) as usize;
            }
        }
        sum
         */
    }
    fn score(&mut self) -> usize {
        if let Some(err) = self.err {
            (1e14 * 0.8_f64.powi(err as i32)
                / ((self.placement_cost() + self.measure_cost()) as f64 + 1e5))
                .ceil() as usize
        } else {
            unreachable!()
        }
    }
    fn noerr_score(&mut self) -> usize {
        (1e14 / ((self.placement_cost() + self.measure_cost()) as f64 + 1e5)).ceil() as usize
    }
    fn placement_cost(&mut self) -> usize {
        if self.placement_cost.is_none() {
            self.placement_cost = Some(match &self.map {
                None => unreachable!(),
                Some(map) => {
                    let mut sum = 0;
                    for i in 0..self.n {
                        for j in 0..self.n {
                            let (px, py) = self.add_pos((i, j), (0, 1));
                            let (qx, qy) = self.add_pos((i, j), (1, 0));
                            let v = map[i][j] as i64;
                            let p = map[px][py] as i64;
                            let q = map[qx][qy] as i64;
                            sum += ((v - p).pow(2) + (v - q).pow(2)) as usize;
                        }
                    }
                    sum
                }
            });
        }
        self.placement_cost.unwrap()
    }

    fn sd_coef(n: usize, m: usize, sd: f64) -> f64 {
        match Classifier(vec![
            (
                vec![
                    (2, false, 36),
                    (1, false, 84),
                    (0, false, 19),
                    (1, false, 83),
                ],
                4,
            ),
            (
                vec![
                    (2, false, 36),
                    (1, false, 84),
                    (0, false, 19),
                    (1, true, 83),
                ],
                3,
            ),
            (
                vec![
                    (2, false, 36),
                    (1, false, 84),
                    (0, true, 19),
                    (1, false, 60),
                ],
                2,
            ),
            (
                vec![(2, false, 36), (1, false, 84), (0, true, 19), (1, true, 60)],
                3,
            ),
            (
                vec![
                    (2, false, 36),
                    (1, true, 84),
                    (1, false, 87),
                    (2, false, 25),
                ],
                2,
            ),
            (
                vec![(2, false, 36), (1, true, 84), (1, false, 87), (2, true, 25)],
                0,
            ),
            (
                vec![(2, false, 36), (1, true, 84), (1, true, 87), (0, false, 11)],
                2,
            ),
            (
                vec![(2, false, 36), (1, true, 84), (1, true, 87), (0, true, 11)],
                4,
            ),
            (
                vec![
                    (2, true, 36),
                    (0, false, 23),
                    (2, false, 49),
                    (0, false, 15),
                ],
                3,
            ),
            (
                vec![(2, true, 36), (0, false, 23), (2, false, 49), (0, true, 15)],
                2,
            ),
            (
                vec![(2, true, 36), (0, false, 23), (2, true, 49), (1, false, 96)],
                1,
            ),
            (
                vec![(2, true, 36), (0, false, 23), (2, true, 49), (1, true, 96)],
                2,
            ),
            (
                vec![(2, true, 36), (0, true, 23), (1, false, 86), (0, false, 25)],
                0,
            ),
            (
                vec![(2, true, 36), (0, true, 23), (1, false, 86), (0, true, 25)],
                0,
            ),
            (
                vec![(2, true, 36), (0, true, 23), (1, true, 86), (2, false, 49)],
                2,
            ),
            (
                vec![(2, true, 36), (0, true, 23), (1, true, 86), (2, true, 49)],
                1,
            ),
        ])
        .pred(n, m, sd, 0, 0, 0)
        {
            0 => 4.,
            1 => 4.5,
            2 => 5.,
            3 => 5.5,
            4 => 6.,
            _ => unreachable!(),
        }
    }

    fn new(
        input: &Input,
        simulation: bool,
        output: bool,
        num_high: usize,
        prob_table: Vec<f64>,
    ) -> Self {
        /*
        let stdin = io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            n: usize,
            m: usize,
            sd: f64,
            poses: [(usize, usize); m],
        }
         */
        let Input {
            n,
            m,
            sd,
            poses,
            answer,
        } = input.clone();
        let mut cum = vec![0.; prob_table.len() + 1];
        for i in 0..prob_table.len() {
            cum[i + 1] = cum[i] + prob_table[i];
        }
        let mut rng = RandXor::new();
        let high_val = 1000.min((sd * Self::sd_coef(n, m, sd)) as usize).max(8);

        Env {
            n,
            m,
            sd,
            turn: 0,
            output,
            poses,
            rng,
            prob_table,
            prob_table_cum: cum,
            logs: vec![vec![]; m],
            answer,
            err: None,
            map: None,
            num_high,
            lb: None,
            placement_cost: None,
            measure_cost: 0,
            high_val,
            second_high: high_val / 4,
            radius: 0,
        }
    }

    fn set_holes2(&mut self, k: usize) -> Vec<(usize, usize)> {
        // TODO: 焼きなましにしたりとか, 実際の値（各出口からの最寄りの距離のsum）を使ったりとかするといい
        let iter_max = 10000;
        let mut map = vec![vec![false; self.n]; self.n];
        let mut holes = vec![(0, 0); k];

        for i in 0..k {
            loop {
                holes[i] = (self.rng.rand_n(self.n), self.rng.rand_n(self.n));
                if holes
                    .iter()
                    .take(i)
                    .find(|&&x| self.che_dist(x, holes[i]) <= 1)
                    .is_none()
                {
                    break;
                }
            }
            map[holes[i].0][holes[i].1] = true;
        }

        let border = 2;
        let border_area = 2 * border * (border + 1) + 1;

        // TODO: ここ多分評価関数が不適切

        /*
        let coef_idx = Classifier(vec![
            (vec![(2, false, 702), (2, true, 110)], 0),
            (vec![(2, false, 30), (2, false, 110)], 1),
            (vec![(2, true, 30), (2, false, 110)], 0),
            (vec![(1, false, 82), (2, true, 702), (2, true, 110)], 0),
            (vec![(1, true, 82), (2, true, 702), (2, true, 110)], 2),
        ])
        .pred(self.n, self.m, self.sd);
         */

        // let coef = [0.3, 0.6, 1.0][coef_idx];

        let coef = if self.sd < 300.0 {
            0.3
        } else if self.sd < 700.0 {
            0.6
        } else {
            1.0
        };

        let cn_fn = |x: &usize| {
            let x = *x;
            assert!(x <= border_area);
            let xc = x.min(border_area - x);
            // (xc as f64).powf(coef)
            (xc as f64).powf(coef)
        };

        let mut cnts = vec![0; self.m];
        for i in 0..self.m {
            for j in 0..k {
                cnts[i] += (self.man_dist(self.poses[i], holes[j]) <= border) as usize;
            }
        }
        let mut score = cnts.iter().map(cn_fn).sum::<f64>();

        // TODO: これ正直score ga改善してるか分からないし不要かも
        /*
        let neighbor_coef = 0.00;
        let neighbors = vec![(0, 1), (1, 0), (0, -1), (-1, 0)];
        for i in 0..k {
            let (x, y) = holes[i];
            for &(dx, dy) in &neighbors {
                let (nx, ny) = self.add_pos((x, y), (dx, dy));
                if map[nx][ny] {
                    score += neighbor_coef;
                }
            }
        }
         */
        let neighbors = vec![
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
        ];
        for _iter in 0..iter_max {
            let i = self.rng.rand_n(k);
            let prev_cnts = cnts.clone();
            let prev = holes[i];
            let nex = (self.rng.rand_n(self.n), self.rng.rand_n(self.n));
            if neighbors
                .iter()
                .find(|&&d| {
                    let p = self.add_pos(nex, d);
                    map[p.0][p.1]
                })
                .is_some()
            {
                continue;
            }
            for j in 0..self.m {
                cnts[j] -= (self.man_dist(self.poses[j], prev) <= border) as usize;
                cnts[j] += (self.man_dist(self.poses[j], nex) <= border) as usize;
            }

            let mut nex_score = cnts.iter().map(cn_fn).sum::<f64>();

            if score <= nex_score {
                holes[i] = nex;
                score = nex_score;
                // println!("{}: {}", iter, score);
                map[prev.0][prev.1] = false;
                map[nex.0][nex.1] = true;
            } else {
                cnts = prev_cnts;
            }
        }
        let dels = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        let holes_tmp = holes.clone();
        for (x, y) in holes_tmp {
            let del_target = self.rng.rand_n(5);
            for i in (0..4).filter(|&x| x != del_target) {
                let (dx, dy) = dels[i];
                let (nx, ny) = self.add_pos((x, y), (dx, dy));
                holes.push((nx, ny));
            }
        }
        holes
    }

    fn set_holes_k1(&mut self) -> Vec<(usize, usize)> {
        let xs = self.poses.iter().map(|&(x, y)| x).collect::<Vec<_>>();
        let ys = self.poses.iter().map(|&(x, y)| y).collect::<Vec<_>>();
        let opt_x = (0..self.n)
            .map(|x| {
                xs.iter()
                    .map(|&x2| (self.man_dist((0, x), (0, x2)) as f64).powf(1.0))
                    .sum::<f64>()
            })
            .enumerate()
            .min_by_key(|x| NotNan::new(x.1).unwrap())
            .unwrap()
            .0;
        let opt_y = (0..self.n)
            .map(|x| {
                ys.iter()
                    .map(|&x2| (self.man_dist((0, x), (0, x2)) as f64).powf(1.0))
                    .sum::<f64>()
            })
            .enumerate()
            .min_by_key(|x| NotNan::new(x.1).unwrap())
            .unwrap()
            .0;
        vec![(opt_x, opt_y)]
    }

    fn set_holes3(&mut self, k: usize) -> Vec<(usize, usize)> {
        let iter_max = 10000;
        let mut map = vec![vec![false; self.n]; self.n];
        let mut holes = vec![(0, 0); k];
        let neighbors = vec![
            (0, 0),
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (2, 0),
            (1, -1),
            (0, -2),
            (-1, -1),
            (-2, 0),
            (-1, 1),
            (0, 2),
            (1, 1),
        ];
        for i in 0..k {
            loop {
                holes[i] = (self.rng.rand_n(self.n), self.rng.rand_n(self.n));
                if holes.iter().take(i).find(|&&x| x == holes[i]).is_none() {
                    break;
                }
            }
            map[holes[i].0][holes[i].1] = true;
        }
        let mut ins = vec![vec![false; self.n]; self.n];
        for &(x, y) in &self.poses {
            ins[x][y] = true;
        }

        let coef = 0.95;

        let mut cnts = vec![0; neighbors.len()];

        for &(x, y) in &holes {
            for (idx, &(dx, dy)) in neighbors.iter().enumerate() {
                let (x, y) = (
                    ((self.n + x) as i64 + dx) as usize % self.n,
                    ((self.n + y) as i64 + dy) as usize % self.n,
                );
                if ins[x][y] {
                    cnts[idx] += 1;
                }
            }
        }

        let mut score = cnts.iter().map(|&x| (x as f64).powf(coef)).sum::<f64>();

        for _iter in 0..iter_max {
            let i = self.rng.rand_n(k);
            let prev_cnts = cnts.clone();
            let prev = holes[i];
            let nex = (self.rng.rand_n(self.n), self.rng.rand_n(self.n));
            if map[nex.0][nex.1] {
                continue;
            }
            {
                let (x, y) = prev;
                for (idx, &(dx, dy)) in neighbors.iter().enumerate() {
                    let (x, y) = (
                        ((self.n + x) as i64 + dx) as usize % self.n,
                        ((self.n + y) as i64 + dy) as usize % self.n,
                    );
                    if ins[x][y] {
                        cnts[idx] -= 1;
                    }
                }
            }
            {
                let (x, y) = nex;
                for (idx, &(dx, dy)) in neighbors.iter().enumerate() {
                    let (x, y) = (
                        ((self.n + x) as i64 + dx) as usize % self.n,
                        ((self.n + y) as i64 + dy) as usize % self.n,
                    );
                    if ins[x][y] {
                        cnts[idx] += 1;
                    }
                }
            }

            let nex_score = cnts.iter().map(|&x| (x as f64).powf(coef)).sum::<f64>();

            if score <= nex_score {
                holes[i] = nex;
                score = nex_score;
                // println!("{}: {}", iter, score);
                map[prev.0][prev.1] = false;
                map[nex.0][nex.1] = true;
            } else {
                cnts = prev_cnts;
            }
        }
        holes
    }

    fn set_holes(&mut self, k: usize) -> Vec<(usize, usize)> {
        // TODO: 焼きなましにしたりとか, 実際の値（各出口からの最寄りの距離のsum）を使ったりとかするといい
        let iter_max = 10000;
        let mut map = vec![vec![false; self.n]; self.n];
        let mut holes = vec![(0, 0); k];
        for i in 0..k {
            loop {
                holes[i] = (self.rng.rand_n(self.n), self.rng.rand_n(self.n));
                if holes.iter().take(i).find(|&&x| x == holes[i]).is_none() {
                    break;
                }
            }
            map[holes[i].0][holes[i].1] = true;
        }

        let border = 2;
        let border_area = 2 * border * (border + 1) + 1;

        let coef = if self.sd < 300.0 {
            0.3
        } else if self.sd < 700.0 {
            0.6
        } else {
            1.0
        };

        let cn_fn = |x: &usize| {
            let x = *x;
            assert!(x <= border_area);
            let xc = x.min(border_area - x);
            // (xc as f64).powf(coef)
            (xc as f64).powf(coef)
        };

        let mut cnts = vec![0; self.m];
        for i in 0..self.m {
            for j in 0..k {
                cnts[i] += (self.man_dist(self.poses[i], holes[j]) <= border) as usize;
            }
        }
        let mut score = cnts.iter().map(cn_fn).sum::<f64>() +
            + holes
            .iter()
            .map(|&p| holes.iter().map(|&q| self.man_dist(p, q)).min().unwrap() as f64)
            .sum::<f64>() * 0.3;

        for _iter in 0..iter_max {
            let i = self.rng.rand_n(k);
            let prev_cnts = cnts.clone();
            let prev = holes[i];
            let nex = (self.rng.rand_n(self.n), self.rng.rand_n(self.n));
            if map[nex.0][nex.1] {
                continue;
            }
            for j in 0..self.m {
                cnts[j] -= (self.man_dist(self.poses[j], prev) <= border) as usize;
                cnts[j] += (self.man_dist(self.poses[j], nex) <= border) as usize;
            }

            let mut nex_score = cnts.iter().map(cn_fn).sum::<f64>();

            if score <= nex_score {
                holes[i] = nex;
                score = nex_score;
                // println!("{}: {}", iter, score);
                map[prev.0][prev.1] = false;
                map[nex.0][nex.1] = true;
            } else {
                cnts = prev_cnts;
            }
        }
        holes
    }

    fn set_holes_smallk(&mut self, k: usize) -> Vec<(usize, usize)> {
        let iter_max = 10000;
        let mut map = vec![vec![false; self.n]; self.n];
        let mut holes = vec![(0, 0); k];
        for i in 0..k {
            loop {
                holes[i] = (self.rng.rand_n(self.n), self.rng.rand_n(self.n));
                if holes.iter().take(i).find(|&&x| x == holes[i]).is_none() {
                    break;
                }
            }
            map[holes[i].0][holes[i].1] = true;
        }

        let mut score = self
            .poses
            .iter()
            .map(|&p| holes.iter().map(|&q| self.man_dist(p, q)).min().unwrap() as f64)
            .sum::<f64>()
            + holes
                .iter()
                .map(|&p| holes.iter().map(|&q| self.man_dist(p, q)).min().unwrap() as f64)
                .sum::<f64>();
        for _iter in 0..iter_max {
            let i = self.rng.rand_n(k);
            let pre = holes[i];
            let nex = (self.rng.rand_n(self.n), self.rng.rand_n(self.n));

            holes[i] = nex;
            let mut nex_score = self
                .poses
                .iter()
                .map(|&p| holes.iter().map(|&q| self.man_dist(p, q)).min().unwrap() as f64)
                .sum::<f64>()
                + holes
                    .iter()
                    .map(|&p| holes.iter().map(|&q| self.man_dist(p, q)).min().unwrap() as f64)
                    .sum::<f64>();

            if score >= nex_score {
                score = nex_score;
                map[pre.0][pre.1] = false;
                map[nex.0][nex.1] = true;
            } else {
                holes[i] = pre;
            }
        }
        holes
    }

    fn solve_small(&mut self, reboot: bool) -> Vec<usize> {
        self.turn = 0;
        self.placement_cost = None;
        self.measure_cost = 0;
        for i in 0..self.m {
            self.logs[i].clear();
        }
        // だいたい0にして, (0, 0) だけ大きな値にする
        // 0 にする個数は後で増やしていい

        let mut preds = vec![0; self.m];

        /* TODO: speed up
        let mut skip = false;
        if let Some(lb) = self.lb {
            let sc_hb = self.high_val.pow(2) * 4 * self.num_high + 1000 * 100;
            if sc_hb <= lb {
                skip = true;
            }
        }
         */

        let (mut map, mut holes) = if !reboot {
            let mut map = vec![vec![0; self.n]; self.n];
            let n_sq = self.n * self.n / 2;

            let mut holes = if self.num_high == 1 {
                self.set_holes_k1()
            } else if Classifier(vec![
                (vec![(2, false, 6), (2, false, 110)], 0),
                (vec![(0, true, 31), (2, true, 6), (2, false, 110)], 0),
                (vec![(1, false, 71), (2, true, 600), (2, true, 110)], 0),
                (vec![(1, true, 71), (2, true, 600), (2, true, 110)], 1),
                (
                    vec![
                        (0, false, 20),
                        (0, false, 31),
                        (2, true, 6),
                        (2, false, 110),
                    ],
                    1,
                ),
                (vec![(1, false, 78), (2, false, 600), (2, true, 110)], 0),
                (
                    vec![
                        (2, false, 462),
                        (1, true, 78),
                        (2, false, 600),
                        (2, true, 110),
                    ],
                    0,
                ),
                (
                    vec![
                        (2, true, 462),
                        (1, true, 78),
                        (2, false, 600),
                        (2, true, 110),
                    ],
                    0,
                ),
                (
                    vec![
                        (2, false, 30),
                        (0, true, 20),
                        (0, false, 31),
                        (2, true, 6),
                        (2, false, 110),
                    ],
                    0,
                ),
                (
                    vec![
                        (2, true, 30),
                        (0, true, 20),
                        (0, false, 31),
                        (2, true, 6),
                        (2, false, 110),
                    ],
                    1,
                ),
            ])
            .pred(
                self.n,
                self.m,
                self.sd,
                self.num_high,
                self.high_val,
                self.radius,
            ) == 1
            {
                if self.num_high <= 3 {
                    self.set_holes_smallk(self.num_high)
                } else if self.num_high == 1 || self.num_high * 4 >= n_sq {
                    self.set_holes3(self.num_high)
                } else {
                    self.set_holes2(self.num_high / 2)
                }
            } else {
                if self.num_high <= 5 {
                    self.set_holes_smallk(self.num_high)
                } else {
                    self.set_holes(self.num_high)
                }
            };

            for &(x, y) in &holes {
                map[x][y] = self.high_val;
            }
            (map, holes)
        } else {
            let mut map = self.map.clone().unwrap();
            let mut holes = vec![];
            let max = map
                .iter()
                .map(|v| v.iter().cloned().max().unwrap())
                .max()
                .unwrap();
            for i in 0..self.n {
                for j in 0..self.n {
                    if map[i][j] == max {
                        map[i][j] = self.high_val;
                        holes.push((i, j));
                    }
                }
            }
            (map, holes)
        };

        let mut move_cands = vec![];
        loop {
            move_cands.clear();
            let mut que = BinaryHeap::new();
            for i in 0..self.m {
                for &hole in &holes {
                    let d = self.man_dist(self.poses[i], hole);
                    let mov = self.get_move(self.poses[i], hole);
                    que.push((1000 - d, mov));
                }
            }
            let mut flags = vec![0u128; self.m];
            let mut while_idx = 0;
            let mut uniq = 1;
            while let Some((_d, (dx, dy))) = que.pop() {
                for i in 0..self.m {
                    let (x, y) = self.add_pos(self.poses[i], (dx, dy));
                    let fl = map[x][y] != 0;
                    flags[i] &= !(1u128 << while_idx);
                    flags[i] |= (fl as u128) << while_idx;
                }
                let mut flags_cp = flags.clone();
                flags_cp.sort();
                let mut cn = 1;
                for i in 0..flags_cp.len() - 1 {
                    cn += (flags_cp[i] != flags_cp[i + 1]) as usize;
                }
                if uniq != cn {
                    uniq = cn;
                    move_cands.push((dx, dy));
                    while_idx += 1;
                }
                if uniq == flags_cp.len() {
                    break;
                }
            }
            if uniq == flags.len() {
                break;
            } else {
                let idx = self.rng.rand_n(holes.len());
                let bef = holes[idx];
                while map[holes[idx].0][holes[idx].1] == self.high_val {
                    if self.rng.rand_n(2) == 0 {
                        holes[idx].0 = (holes[idx].0 + 1) % self.n;
                    } else {
                        holes[idx].1 = (holes[idx].1 + 1) % self.n;
                    }
                }
                let aft = holes[idx];
                map[aft.0][aft.1] = self.high_val;
                map[bef.0][bef.1] = 0;
            }
        }

        self.second_high = match Classifier(vec![
            (
                vec![
                    (2, false, 576),
                    (0, false, 38),
                    (0, false, 18),
                    (3, false, 1),
                ],
                2,
            ),
            (
                vec![
                    (2, false, 576),
                    (0, false, 38),
                    (0, false, 18),
                    (3, true, 1),
                ],
                4,
            ),
            (
                vec![
                    (2, false, 576),
                    (0, false, 38),
                    (0, true, 18),
                    (3, false, 153),
                ],
                4,
            ),
            (
                vec![
                    (2, false, 576),
                    (0, false, 38),
                    (0, true, 18),
                    (3, true, 153),
                ],
                0,
            ),
            (
                vec![
                    (2, false, 576),
                    (0, true, 38),
                    (3, false, 15),
                    (5, false, 1),
                ],
                4,
            ),
            (
                vec![(2, false, 576), (0, true, 38), (3, false, 15), (5, true, 1)],
                5,
            ),
            (
                vec![
                    (2, false, 576),
                    (0, true, 38),
                    (3, true, 15),
                    (3, false, 20),
                ],
                4,
            ),
            (
                vec![(2, false, 576), (0, true, 38), (3, true, 15), (3, true, 20)],
                3,
            ),
            (
                vec![
                    (2, true, 576),
                    (0, false, 31),
                    (5, false, 7),
                    (0, false, 13),
                ],
                0,
            ),
            (
                vec![(2, true, 576), (0, false, 31), (5, false, 7), (0, true, 13)],
                1,
            ),
            (
                vec![(2, true, 576), (0, false, 31), (5, true, 7), (1, false, 63)],
                3,
            ),
            (
                vec![(2, true, 576), (0, false, 31), (5, true, 7), (1, true, 63)],
                2,
            ),
            (
                vec![
                    (2, true, 576),
                    (0, true, 31),
                    (2, false, 676),
                    (0, false, 48),
                ],
                2,
            ),
            (
                vec![
                    (2, true, 576),
                    (0, true, 31),
                    (2, false, 676),
                    (0, true, 48),
                ],
                1,
            ),
            (
                vec![(2, true, 576), (0, true, 31), (2, true, 676), (3, false, 5)],
                0,
            ),
            (
                vec![(2, true, 576), (0, true, 31), (2, true, 676), (3, true, 5)],
                4,
            ),
        ])
        .pred(
            self.n,
            self.m,
            self.sd,
            self.num_high,
            self.high_val,
            self.radius,
        ) {
            0 => (self.high_val as f64 * 0.15) as usize,
            1 => (self.high_val as f64 * 0.20) as usize,
            2 => (self.high_val as f64 * 0.25) as usize,
            3 => (self.high_val as f64 * 0.30) as usize,
            4 => (self.high_val as f64 * 0.35) as usize,
            5 => (self.high_val as f64 * 0.40) as usize,
            _ => unreachable!(),
        };

        {
            let mut v = holes.clone();
            let mut value = self.second_high;
            for _r in 0..self.radius {
                // TODO: これ不正確で, 例えばr=2のときは (1,1)=250/2=125, (2,0)=250/4=62.5 みたいにした方がいい
                let mut w = vec![];
                for (x, y) in v {
                    for (dx, dy) in vec![(0, 1), (1, 0), (0, -1), (-1, 0)] {
                        let (nx, ny) = self.add_pos((x, y), (dx, dy));
                        if map[nx][ny] == 0 {
                            map[nx][ny] = value;
                            w.push((nx, ny));
                        }
                    }
                }
                v = w;
                // value = (value as f64 * 0.5) as usize;
                value = (value as f64 * 0.6) as usize;
            }
        }
        /*
        {
            let mut v = holes.clone();
            let mut poses = vec![];
            for _r in 0..self.radius {
                // TODO: これ不正確で, 例えばr=2のときは (1,1)=250/2=125, (2,0)=250/4=62.5 みたいにした方がいい
                let mut w = vec![];
                for (x, y) in v {
                    for (dx, dy) in vec![(0, 1), (1, 0), (0, -1), (-1, 0)] {
                        let (nx, ny) = self.add_pos((x, y), (dx, dy));
                        if map[nx][ny] == 0 {
                            poses.push((nx, ny));
                        }
                        w.push((nx, ny));
                    }
                }
                v = w;
            }
            poses.sort();
            let mut v = vec![];
            for i in 1..poses.len() {
                if i == 0 || poses[i - 1] != poses[i] {
                    v.push(poses[i]);
                }
            }
            for iter in 0..10 {
                for &(x, y) in &v {
                    let sum = vec![(0, 1), (1, 0), (0, -1), (-1, 0)]
                        .iter()
                        .map(|&(dx, dy)| {
                            let (nx, ny) = self.add_pos((x, y), (dx, dy));
                            map[nx][ny]
                        })
                        .sum::<usize>();
                    map[x][y] = (sum / 4).min(self.high_val / 4);
                }
            }
        }
         */

        if self.output {
            println!("# {}", self.num_high);
            println!("# {}", self.high_val);
            println!("# {}", self.radius);
            println!("# {}", self.second_high);
        }
        self.output_map(&map);

        /*
        let mut bs = BTreeSet::new();
        let mut shortest_outs = self
            .poses
            .iter()
            .map(|&p| {
                let ret = holes
                    .iter()
                    .filter(|&&q| !bs.contains(&self.get_move(p, q)))
                    .map(|&q| (self.man_dist(p, q), q))
                    .min()
                    .map(|(x, y)| y)
                    .unwrap();
                let mov = self.get_move(p, ret);
                bs.insert(mov);
                ret
            })
            .collect::<Vec<_>>();
        let move_cands = self
            .poses
            .iter()
            .zip(shortest_outs.iter())
            .map(|(&x, &y)| self.get_move(x, y))
            .collect::<Vec<_>>();
         */

        // move_candsの各操作をしたときの値が0かを各穴に対して求めている
        let mut move_actuals = vec![vec![false; self.m]; move_cands.len()];
        for i in 0..move_cands.len() {
            for j in 0..self.m {
                let (x, y) = self.add_pos(self.poses[j], move_cands[i]);
                move_actuals[i][j] = (map[x][y] >= self.high_val / 2);
            }
        }

        let mut in_determined = vec![false; self.m];
        let mut out_determined = vec![false; self.m];
        let mut hole_break = false;

        let border_prob_ln = match Classifier(vec![
            (vec![(3, false, 2), (0, false, 25), (0, false, 14)], 4),
            (vec![(3, false, 2), (0, false, 25), (0, true, 14)], 5),
            (vec![(3, false, 2), (0, true, 25), (1, false, 70)], 5),
            (vec![(3, false, 2), (0, true, 25), (1, true, 70)], 4),
            (vec![(3, true, 2), (3, false, 42), (1, false, 97)], 3),
            (vec![(3, true, 2), (3, false, 42), (1, true, 97)], 2),
            (vec![(3, true, 2), (3, true, 42), (3, false, 90)], 2),
            (vec![(3, true, 2), (3, true, 42), (3, true, 90)], 0),
        ])
        .pred(
            self.n,
            self.m,
            self.sd,
            self.num_high,
            self.high_val,
            self.radius,
        ) {
            0 => 5.0,
            1 => 6.0,
            2 => 7.0,
            3 => 8.0,
            4 => 9.0,
            5 => 10.0,
            _ => unreachable!(),
        };

        let pred_class = Classifier(vec![
            (vec![(3, false, 1)], 0),
            (
                vec![
                    (1, false, 83),
                    (3, false, 2),
                    (3, false, 47),
                    (1, true, 74),
                    (3, true, 1),
                ],
                1,
            ),
            (
                vec![
                    (1, true, 83),
                    (3, false, 2),
                    (3, false, 47),
                    (1, true, 74),
                    (3, true, 1),
                ],
                0,
            ),
            (
                vec![
                    (0, false, 40),
                    (3, true, 2),
                    (3, false, 47),
                    (1, true, 74),
                    (3, true, 1),
                ],
                1,
            ),
            (
                vec![
                    (0, true, 40),
                    (3, true, 2),
                    (3, false, 47),
                    (1, true, 74),
                    (3, true, 1),
                ],
                1,
            ),
            (
                vec![(0, false, 48), (3, true, 47), (1, true, 74), (3, true, 1)],
                0,
            ),
            (
                vec![(0, true, 48), (3, true, 47), (1, true, 74), (3, true, 1)],
                1,
            ),
            (vec![(0, false, 22), (1, false, 74), (3, true, 1)], 0),
            (
                vec![(3, false, 61), (0, true, 22), (1, false, 74), (3, true, 1)],
                0,
            ),
            (
                vec![(3, true, 61), (0, true, 22), (1, false, 74), (3, true, 1)],
                0,
            ),
        ])
        .pred(
            self.n,
            self.m,
            self.sd,
            self.num_high,
            self.high_val,
            self.radius,
        );
        if self.output && pred_class == 1 {
            let get_selected_idxes = |log_probs: &Vec<f64>, cands: &Vec<usize>| {
                let mut sorted_log_probs =
                    cands.iter().map(|&i| (i, log_probs[i])).collect::<Vec<_>>();
                sorted_log_probs.sort_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap().reverse());
                let prob_max = sorted_log_probs[0].1;
                // let border = prob_max - border_prob_ln;
                let selection_cnt = 3;
                // 尤度の上位 selection_cnt 個
                let mut border =
                    sorted_log_probs[(selection_cnt - 1).min(sorted_log_probs.len() - 1)].1;
                if border < prob_max - border_prob_ln {
                    border = prob_max - border_prob_ln;
                }
                let mut selected_idxes = vec![];
                for &(i, val) in &sorted_log_probs {
                    if val >= border {
                        selected_idxes.push(i);
                    } else {
                        break;
                    }
                }
                selected_idxes
            };
            let calc_value = |move_idx: usize, selected_idxes: &Vec<usize>| {
                let v = &move_actuals[move_idx];
                let mut cn = selected_idxes.iter().map(|&j| v[j] as usize).sum::<usize>();
                // これが小さいほど偏っている
                let kat = cn.min(selected_idxes.len() - cn) as f64 / selected_idxes.len() as f64;
                // この値が大きいほど選ばれやすい
                if kat == 0.0 {
                    -1.0
                } else {
                    kat - 0.001
                        * (move_cands[move_idx].0.abs() + move_cands[move_idx].1.abs()) as f64
                }
            };

            let mut log_probs = vec![vec![0.0; self.m]; self.m];
            loop {
                // strategy
                let cands = (0..self.m)
                    .filter(|&x| !out_determined[x])
                    .collect::<Vec<_>>();
                if cands.len() == 1 {
                    break;
                }
                // que<(f64, usize,usize)>:  (value, in_idx, move_idx)
                let mut que = BinaryHeap::new();
                for i in 0..self.m {
                    // TODO: get selected_idxes
                    if !in_determined[i] {
                        let selected_idxes = get_selected_idxes(&log_probs[i], &cands);
                        for j in 0..move_actuals.len() {
                            let sc = calc_value(j, &selected_idxes);
                            que.push((NotNan::new(sc).unwrap(), i, j));
                        }
                    }
                }
                // println!("# top: {}", *que.peek().unwrap().0);
                for iter in 0.. {
                    let (que_sc, hole_idx, move_idx) = que.pop().unwrap();
                    let selected_idxes = get_selected_idxes(&log_probs[hole_idx], &cands);
                    let sc = calc_value(move_idx, &selected_idxes);
                    if in_determined[hole_idx] {
                        continue;
                    }
                    if (*que_sc - sc).abs() > 1e-6 {
                        continue;
                    }
                    let (dx, dy) = move_cands[move_idx];
                    let value = self.sample(&map, hole_idx, (dx, dy));

                    // println!("#: {}:    * / {}", *que_sc, selected_idxes.len());

                    for &out_cand in &cands {
                        let (x, y) = self.add_pos(self.poses[out_cand], (dx, dy));
                        let prob = self.probability(map[x][y], value);
                        log_probs[hole_idx][out_cand] += if prob > 1e-8 { prob.ln() } else { -1e9 };
                    }
                    let selected_idxes = get_selected_idxes(&log_probs[hole_idx], &cands);
                    // println!("#: next -> {}", selected_idxes.len());
                    let (max_idx, max) = log_probs[hole_idx]
                        .iter()
                        .cloned()
                        .enumerate()
                        .max_by(|(xi, x), (yi, y)| x.partial_cmp(y).unwrap())
                        .unwrap();
                    // softmax
                    let normalized = log_probs[hole_idx]
                        .iter()
                        .map(|&x| x - max)
                        .collect::<Vec<_>>();
                    let diff = normalized
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, v)| {
                            if idx == max_idx {
                                None
                            } else {
                                Some((*v).abs())
                            }
                        })
                        .min_by(|x, y| x.partial_cmp(y).unwrap())
                        .unwrap();
                    // println!("# kakutei: {}", diff);
                    // if diff >= border_prob_ln || selected_idxes.len() == 1 {
                    if diff >= border_prob_ln {
                        // println!("# OK turn {}: {} -> {}", self.turn, hole_idx, max_idx);
                        preds[hole_idx] = max_idx;
                        in_determined[hole_idx] = true;
                        out_determined[max_idx] = true;
                        for i in 0..self.m {
                            log_probs[hole_idx][i] = -1e18;
                            log_probs[i][max_idx] = -1e18;
                        }
                        break;
                    }
                    for move_idx in 0..move_actuals.len() {
                        let sc = calc_value(move_idx, &selected_idxes);
                        que.push((NotNan::new(sc).unwrap(), hole_idx, move_idx));
                    }
                    if let Some(lb) = self.lb {
                        if self.err.is_some() && self.score() <= lb {
                            break;
                        }
                    }
                    if self.turn == 10000 {
                        break;
                    }
                }
                if self.turn == 10000 {
                    break;
                }
            }
        } else {
            for hole in 0..self.m {
                // eprintln!();
                // eprintln!("{}: {}", hole, self.turn);
                // strategy
                let cands = (0..self.m)
                    .filter(|&x| !out_determined[x])
                    .collect::<Vec<_>>();

                if cands.len() == 1 {
                    preds[hole] = cands[0];
                    out_determined[cands[0]] = true;
                    break;
                }

                let mut log_probs = vec![0.0; cands.len()];
                for iter in 0.. {
                    // TODO: partial sort or use heap
                    let mut sorted_log_probs = log_probs
                        .iter()
                        .zip(cands.iter())
                        .map(|(x, y)| (*y, *x))
                        .collect::<Vec<_>>();
                    sorted_log_probs.sort_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap().reverse());
                    let prob_max = sorted_log_probs[0].1;
                    // let border = prob_max - border_prob_ln;
                    let selection_cnt = 3;
                    // 尤度の上位 selection_cnt 個
                    let mut border =
                        sorted_log_probs[(selection_cnt - 1).min(sorted_log_probs.len() - 1)].1;
                    if border < prob_max - border_prob_ln {
                        border = prob_max - border_prob_ln;
                    }
                    let mut selected_idxes = vec![];
                    for &(i, val) in &sorted_log_probs {
                        if val >= border {
                            selected_idxes.push(i);
                        } else {
                            break;
                        }
                    }
                    /*
                    if self.output && self.turn <= 50 {
                        println!(
                            "# {:?}",
                            sorted_log_probs
                                .iter()
                                .take(selected_idxes.len())
                                .collect::<Vec<_>>()
                        );
                    }
                     */
                    // selected_idxes.resize(selection_cnt.min(selected_idxes.len()), 0);
                    self.rng.shuffle(&mut selected_idxes);

                    let (target_move_idx, kat) = move_actuals
                        .iter()
                        .enumerate()
                        .map(|(idx, v)| {
                            let mut cn =
                                selected_idxes.iter().map(|&j| v[j] as usize).sum::<usize>();
                            // これが小さいほど偏っている
                            let kat = cn.min(selected_idxes.len() - cn);
                            (
                                idx,
                                kat as f64
                                    - 0.001
                                        * (move_cands[idx].0.abs() + move_cands[idx].1.abs())
                                            as f64,
                            )
                        })
                        .max_by(|&(_idx, cnt), &(_idx2, cnt2)| cnt.partial_cmp(&cnt2).unwrap())
                        .unwrap();

                    let mut cn = selected_idxes
                        .iter()
                        .map(|&j| move_actuals[target_move_idx][j] as usize)
                        .sum::<usize>();
                    if iter == 0 && cn <= 1 {
                        if self.output {
                            println!("# {:?}", move_cands[target_move_idx]);
                            println!(
                                "# {:?} {}",
                                selected_idxes,
                                cn.min(selected_idxes.len() - cn)
                            );
                        }
                        hole_break = true;
                        break;
                    }

                    if self.output {
                        println!("#kat: {} / {}", kat, selected_idxes.len());
                    }

                    let (dx, dy) = move_cands[target_move_idx];

                    let value = self.sample(&map, hole, (dx, dy));

                    for (idx, &out_cand) in cands.iter().enumerate() {
                        let (x, y) = self.add_pos(self.poses[out_cand], (dx, dy));
                        let prob = self.probability(map[x][y], value);
                        log_probs[idx] += if prob > 1e-8 { prob.ln() } else { -1e9 };
                    }
                    let (max_idx, max) = log_probs
                        .iter()
                        .enumerate()
                        .max_by(|(xi, x), (yi, y)| x.partial_cmp(y).unwrap())
                        .map(|(x, y)| (x, *y))
                        .unwrap();
                    // softmax
                    let normalized = log_probs.iter().map(|&x| x - max).collect::<Vec<_>>();
                    let diff = normalized
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, v)| {
                            if idx == max_idx {
                                None
                            } else {
                                Some((*v).abs())
                            }
                        })
                        .min_by(|x, y| x.partial_cmp(y).unwrap())
                        .unwrap();
                    if diff >= border_prob_ln {
                        preds[hole] = cands[max_idx];
                        in_determined[hole] = true;
                        out_determined[cands[max_idx]] = true;
                        break;
                    }
                    if let Some(lb) = self.lb {
                        if self.err.is_some() && self.score() <= lb {
                            break;
                        }
                    }
                    if self.turn == 10000 {
                        break;
                    }
                }
                if let Some(answer) = self.answer.as_ref() {
                    let correct = answer[hole] == preds[hole];
                    self.err = Some(self.err.unwrap_or(0) + !correct as usize);
                }
                if let Some(lb) = self.lb {
                    if self.err.is_some() && self.score() <= lb {
                        break;
                    }
                }
                if self.turn == 10000 || hole_break {
                    break;
                }
            }
        }
        // ここからは距離が近い順に決めていく
        if hole_break {
            while in_determined.iter().any(|&x| !x) {
                let in_cands = (0..self.m)
                    .filter(|&x| !in_determined[x])
                    .collect::<Vec<_>>();
                if in_cands.len() == 1 {
                    break;
                }
                let out_cands = (0..self.m)
                    .filter(|&x| !out_determined[x])
                    .collect::<Vec<_>>();

                let mut dxdys = vec![];
                for &idx in &out_cands {
                    for &hole in &holes {
                        dxdys.push(self.get_move(self.poses[idx], hole));
                    }
                }
                dxdys.sort();
                let mut dxdy_cands = vec![];
                for i in 0..dxdys.len() {
                    if i != 0 && dxdys[i] == dxdys[i - 1] {
                        continue;
                    }
                    // hole
                    let (dx, dy) = dxdys[i];
                    let mut high_cnt = 0;
                    let mut high = 0;
                    let mut low = 0;
                    for &out_idx in &out_cands {
                        let (x, y) = self.add_pos(self.poses[out_idx], (dx, dy));
                        if map[x][y] == self.high_val {
                            high = out_idx;
                            high_cnt += 1;
                        } else {
                            low = out_idx;
                        }
                    }
                    if high_cnt == 1 {
                        dxdy_cands.push((high, dxdys[i]));
                    }
                }
                let op = dxdy_cands
                    .iter()
                    .cloned()
                    .min_by_key(|&(idx, (x, y))| x.abs() + y.abs());
                assert!(op.is_some());
                if op.is_none() {}
                let (out_idx, (dx, dy)) = op.unwrap();
                // let (out_idx_, (dx_, dy_)) = op.unwrap();
                /*
                let (out_idx, (dx, dy)) = out_cands
                    .iter()
                    .map(|&idx| {
                        (
                            idx,
                            holes
                                .iter()
                                .map(|&hole| self.get_move(self.poses[idx], hole))
                                .min_by_key(|&(x, y)| x.abs() + y.abs())
                                .unwrap(),
                        )
                    })
                    .min_by_key(|&(idx, (x, y))| x.abs() + y.abs())
                    .unwrap();
                 */

                /*
                if (out_idx, (dx, dy)) != (out_idx_, (dx_, dy_)) && self.output {
                    println!("# {:?}", (out_idx, (dx, dy)));
                    println!("# {:?}", (out_idx_, (dx_, dy_)));
                    println!("# {} {}", out_cands.len(), dxdy_cands.len());
                }
                 */

                let low = out_cands
                    .iter()
                    .filter(|&&idx| idx != out_idx)
                    .map(|&idx| {
                        let (x, y) = self.add_pos(self.poses[idx], (dx, dy));
                        map[x][y]
                    })
                    .max()
                    .unwrap();

                // poses[out_idx] + (dx, dy) がhighになっていて, それ以外が <= lowなはず
                // 各 i について (i, dx, dy) でクエリを投げ続け, この値が高かったら i が答え
                let mut log_probs = vec![0.0; in_cands.len()];
                let mut last_turn = self.turn;
                loop {
                    if in_cands.len() == 1 {
                        let max_idx = 0;
                        preds[in_cands[max_idx]] = out_idx;
                        in_determined[in_cands[max_idx]] = true;
                        out_determined[out_idx] = true;
                        break;
                    }
                    let idx = log_probs
                        .iter()
                        .enumerate()
                        .max_by(|&(idx, pro), &(idx2, pro2)| pro.partial_cmp(pro2).unwrap())
                        .unwrap()
                        .0;
                    if self.output {
                        // println!("# cands: {}", in_cands.len());
                        println!("# max prob: {}", log_probs[idx]);
                        // println!("# in cands: {:?}", in_cands);
                        // println!("# out cands: {:?}", out_cands);
                    }
                    let i = in_cands[idx];
                    let value = self.sample(&map, i, (dx, dy));
                    // 確定のためには確率最大のものと2番目の差を見て, それが一定以上になればいい
                    // 確率最大=1000 なので, 確率が2番目に高いものも2番目に高い値になる
                    let prob_high = self.probability(self.high_val, value);
                    let prob_low = self.probability(low, value); // TODO: use second highest
                    log_probs[idx] += if prob_high > 1e-8 {
                        prob_high.ln()
                    } else {
                        -1e9
                    };
                    log_probs[idx] -= if prob_low > 1e-8 { prob_low.ln() } else { -1e9 };

                    // TODO: koko set suru
                    let (max_idx, prob_max) = log_probs
                        .iter()
                        .enumerate()
                        .max_by(|&(i, x), &(j, y)| x.partial_cmp(y).unwrap())
                        .unwrap();
                    let prob_second_max = log_probs
                        .iter()
                        .enumerate()
                        .filter(|&(i, x)| i != max_idx)
                        .max_by(|&(i, x), &(j, y)| x.partial_cmp(y).unwrap())
                        .unwrap()
                        .1
                        .clone();
                    let log_pr_diff = prob_max - prob_second_max;
                    if log_pr_diff >= border_prob_ln || self.turn - last_turn >= 1000 {
                        preds[in_cands[max_idx]] = out_idx;
                        in_determined[in_cands[max_idx]] = true;
                        out_determined[out_idx] = true;
                        last_turn = self.turn;
                        if self.output {
                            println!("# {} -> {}", in_cands[max_idx], out_idx);
                        }
                        break;
                    }
                    if self.turn == 10000 {
                        break;
                    }
                }
                if self.turn == 10000 {
                    break;
                }
            }
        }
        let mut used = vec![false; self.m];
        for i in 0..self.m {
            if in_determined[i] {
                used[preds[i]] = true;
            }
        }
        for i in 0..self.m {
            if in_determined[i] {
                continue;
            }
            if !used[preds[i]] {
                used[preds[i]] = true;
            } else {
                let idx = (0..self.m).find(|&i| !used[i]).unwrap();
                preds[i] = idx;
                used[idx] = true;
            }
        }
        preds
    }
    fn solve(&mut self, reboot: bool) {
        let preds = self.solve_small(reboot);
        self.output_pred(&preds);
        if let Some(answer) = self.answer.as_ref() {
            let mut cnt = 0;
            if self.turn == 10000 {
                eprintln!("turn limit exceeded");
            }
            for i in 0..self.m {
                if answer[i] != preds[i] {
                    if self.turn != 10000 {
                        eprintln!(
                            "wrong {}: actual pred: {:?} {:?}     : [{},{}]",
                            i, self.poses[answer[i]], self.poses[preds[i]], answer[i], preds[i],
                        );
                    }
                    cnt += 1;
                }
            }
            self.err = Some(cnt);
            eprintln!("turn: {}", self.turn);
            eprintln!("place   cost: {}", self.placement_cost());
            eprintln!("measure cost: {}", self.measure_cost());
            eprintln!("error: {}", cnt);
            eprintln!("score: {}", self.score());
            // eprintln!("noerror score: {}", self.noerr_score());
        }
    }
}

struct Optimizer {
    input: Input,
    rng: RandXor,
    prob_table: Vec<f64>,
    cnt: usize,
    lb: usize,
    envs: BTreeMap<usize, Env>,
    start_time: Instant,
}

impl Optimizer {
    const END_TIME: u128 = 3700;
    fn new(input: &Input, start_time: Instant, prob_table: Vec<f64>) -> Self {
        let mut answer = (0..input.m).collect::<Vec<_>>();
        let mut rng = RandXor::new();
        rng.shuffle(&mut answer);
        let mut input = input.clone();
        input.answer = Some(answer.clone());
        Self {
            input,
            rng,
            prob_table: prob_table.clone(),
            cnt: 0,
            lb: 0,
            envs: BTreeMap::new(),
            start_time,
        }
    }
    fn check(&mut self, num_high: usize, use_lb: bool) -> (usize, usize) {
        self.cnt += 1;
        let mut env = Env::new(&self.input, true, false, num_high, self.prob_table.clone());
        if use_lb {
            env.lb = Some(self.lb);
        }
        eprintln!("num high: {}", num_high);
        env.solve(false);
        eprintln!();
        let sc = if env.turn >= 9000 { 0 } else { env.score() };
        // let sc = (env.noerr_score() as f64 * 0.2_f64.powi(env.err.unwrap() as i32)).round() as usize;
        if use_lb {
            self.lb = self.lb.max(sc);
        }
        self.envs.insert(num_high, env);
        (sc, num_high)
    }
    fn check_2(&mut self, num_high: usize) -> (usize, usize) {
        let min_sc = (0..2)
            .map(|_| {
                self.reset_answer();
                self.check(num_high, false).0 as f64
            })
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        (min_sc as usize, num_high)
        // (sc1.min(sc2).min(sc3) as usize, num_high)
        // ((sc1 * sc2 * sc3).sqrt() as usize, num_high)
    }
    fn reset_answer(&mut self) {
        let mut input = self.input.clone();
        let mut answer = (0..input.m).collect::<Vec<_>>();
        self.rng.shuffle(&mut answer);
        input.answer = Some(answer.clone());
        self.input = input;
    }
    fn find_num_high(&mut self) -> (usize, usize) {
        let n_sq = self.input.n * self.input.n;
        let max_high = (n_sq / 2).min(250);
        let skip = (max_high / 25).max(1);

        let filtered_max = (1..max_high)
            .filter(|&i| i <= 5 || i + 1 >= max_high || i % skip == 0)
            .map(|i| self.check(i, true))
            .max()
            .unwrap()
            .1;
        self.lb = 0;
        let skip2 = if filtered_max <= 5 {
            1
        } else {
            (skip as f64).sqrt() as usize
        };
        let filtered_max2 = (filtered_max as i64 - skip as i64..=filtered_max as i64 + skip as i64)
            .filter_map(|i| {
                if 1 <= i && i < max_high as i64 && (i % skip2 as i64 == 0) {
                    Some(self.check(i as usize, true))
                } else {
                    None
                }
            })
            .max()
            .unwrap()
            .1;
        self.lb = 0;
        let skip3 = skip2.max(3);
        (filtered_max2 as i64 - skip3 as i64..=filtered_max2 as i64 + skip3 as i64)
            .filter_map(|i| {
                if 1 <= i && i < max_high as i64 {
                    Some(self.check_2(i as usize))
                } else {
                    None
                }
            })
            // .map(|i| self.check(i, true))
            .max()
            .unwrap()
    }
    fn find_high_val(&mut self, num_high: usize, radius: usize) -> usize {
        assert!(self.envs.contains_key(&num_high));
        let mut env = self.envs.get(&num_high).unwrap().clone();
        if self.start_time.elapsed().as_millis() >= Self::END_TIME {
            return env.high_val;
        }
        env.radius = radius;
        env.solve(false);
        let mut rng = env.rng.clone();

        let mut sc = env.score();
        let mut high_val = env.high_val;
        // if env.num_high > 10 || env.score() <= 10000000 || env.turn >= 8000 {
        if env.turn >= 8000 {
            return high_val;
        }

        let mut ma = sc;
        let max_val = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            .iter()
            .rev()
            .map(|&rate| {
                if self.start_time.elapsed().as_millis() >= Self::END_TIME {
                    return (0, env.high_val);
                }
                env.rng = rng.clone();
                env.high_val = (high_val as f64 * rate) as usize;
                if env.high_val <= 1 {
                    (0, env.high_val)
                } else if rate == 1.0 {
                    (sc, env.high_val)
                } else {
                    eprintln!();
                    eprintln!("high val: {}", env.high_val);
                    env.lb = Some(ma);
                    env.solve(true);
                    eprintln!();
                    let new_sc = if env.turn >= 9000 { 0 } else { env.score() };
                    ma = ma.max(new_sc);
                    ((new_sc as f64 * 1.0) as usize, env.high_val)
                }
            })
            .max()
            .unwrap()
            .1;
        // high_val
        max_val
    }
    fn find_second_high(&mut self, num_high: usize, radius: usize, high_val: usize) -> usize {
        assert!(self.envs.contains_key(&num_high));
        let mut env = self.envs.get(&num_high).unwrap().clone();
        if self.start_time.elapsed().as_millis() >= Self::END_TIME {
            return env.high_val;
        }
        env.radius = radius;
        env.high_val = high_val;
        env.solve(false);
        let mut rng = env.rng.clone();

        let mut sc = env.score();

        let mut ma = sc;
        self.lb = 0;
        let max_val = vec![0.2, 0.25, 0.3]
            .iter()
            .rev()
            .map(|&rate| {
                if self.start_time.elapsed().as_millis() >= Self::END_TIME {
                    return (0, env.second_high);
                }
                env.rng = rng.clone();
                env.second_high = ((high_val as f64 * rate) as usize).min((high_val - 1) / 2);
                if env.second_high <= 1 {
                    (0, env.second_high)
                } else {
                    eprintln!();
                    eprintln!("second high: {}", env.second_high);
                    env.lb = Some(ma);
                    env.solve(true);
                    eprintln!();
                    let new_sc = if env.turn >= 9000 { 0 } else { env.score() };
                    ma = ma.max(new_sc);
                    ((new_sc as f64 * 1.0) as usize, env.second_high)
                }
            })
            .max()
            .unwrap()
            .1;
        // high_val
        max_val
    }
    fn find_radius(&mut self, num_high: usize, high_val: usize) -> usize {
        let mut env = self.envs.get(&num_high).unwrap().clone();
        env.high_val = high_val;
        let mut best = (env.score(), 0);
        let mut high_4 = high_val / 4;
        for r in 1..10 {
            if self.start_time.elapsed().as_millis() >= Self::END_TIME {
                break;
            }
            if high_4 == 0 {
                break;
            } else {
                high_4 /= 2;
            }
            let mut env = env.clone();
            env.lb = Some(best.0);
            env.radius = r;
            env.solve(true);
            let new_sc = if env.turn >= 8000 { 0 } else { env.score() };
            let cand = (new_sc, r);
            if best < cand {
                best = cand;
            }
            /* else if best.1 + 1 < r {
                break;
            }*/
        }
        best.1
    }
}

fn main() {
    let start = Instant::now();
    let simulate = std::env::var("simulate").is_ok();
    let input = Input::new(simulate);

    let prob_table = make_prob_table(input.sd);

    let mut opt = Optimizer::new(&input, start, prob_table.clone());
    let num_high = opt.find_num_high().1;
    // let high_val = opt.find_high_val(num_high, radius);
    let radius = opt.find_radius(
        num_high,
        1000.min((input.sd * Env::sd_coef(input.n, input.m, input.sd)) as usize)
            .max(8),
    );
    let high_val = opt.find_high_val(num_high, radius);
    // let second_high = opt.find_second_high(num_high, radius, high_val);

    let mut env = if simulate {
        Env::new(&input, simulate, false, num_high, prob_table.clone())
    } else {
        Env::new(&input, simulate, true, num_high, prob_table.clone())
    };
    eprintln!();
    eprintln!("run cnt: {}", opt.cnt);
    eprintln!("opt num_high: {}", num_high);
    eprintln!("opt high_val: {}", high_val);
    // eprintln!("opt second_high: {}", second_high);
    eprintln!("opt radius: {}", radius);
    env.radius = radius;
    env.high_val = high_val;
    env.solve(false);
}
