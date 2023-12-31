cargo build --manifest-path=./rs/Cargo.toml --release 2> /dev/null
time cargo run --manifest-path=./tools/Cargo.toml --release --bin tester ./rs/target/release/rs < ./tools/in/$1.txt > ./tools/out/$1.txt
cat ./tools/out/$1.txt | xclip -selection c
