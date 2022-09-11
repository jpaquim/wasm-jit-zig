# wasm-jit-zig

Build with:

```sh
zig build -Dtarget=wasm32-freestanding
```

For production:

```sh
zig build -Dtarget=wasm32-freestanding -Drelease-small
```
