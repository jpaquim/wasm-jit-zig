# wasm-jit-zig

Build with:

```sh
zig build-lib -dynamic -target wasm32-freestanding src/main.zig --name interplib && cp interplib.wasm public/
```

For production:

```sh
zig build-lib -O ReleaseSmall -dynamic -target wasm32-freestanding src/main.zig --name interplib && cp interplib.wasm public/
```
