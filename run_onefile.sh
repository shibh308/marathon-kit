path=out/out$2.txt
./cpp/cmake-build-release/cpp < $1 > $path
cargo run --manifest-path=./tools/Cargo.toml --release --bin vis $1 $path
