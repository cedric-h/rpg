cargo build --target=wasm32-unknown-unknown --release
cp ./target/wasm32-unknown-unknown/release/rpg.wasm ./rpg.wasm
wasm-strip rpg.wasm
