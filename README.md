# `safetensors-c` — Embeddable C Library Wrapper for Rust `safetensors`

`safetensors-c` wraps the official Hugging Face [`safetensors`](https://github.com/safetensors/safetensors) Rust crate (`v0.8`) into a standalone static library (`libsafetensors_c.a` / `safetensors_c.lib`) paired with a single embeddable C/C++ header file (`include/safetensors.h`).

## Key Capabilities

- **Direct Rust Crate Wrapper**: Delegates header validation, zero-copy tensor slicing, and serialization directly to `safetensors::SafeTensors`, `safetensors::tensor::TensorView`, `safetensors::serialize`, and `safetensors::serialize_to_file`.
- **Cross-Platform Zero-Copy Mmap**: Uses [`memmap2`](https://crates.io/crates/memmap2) (`mmap` on Linux/macOS and `CreateFileMappingW` / `MapViewOfFile` on Windows) so tensor data pointers (`SafetensorsTensorView.data`) point directly into OS-mapped pages without extra heap allocations.
- **Full Windows Support**:
  - Works with both **MinGW-w64 / GNU** (`x86_64-pc-windows-gnu`, producing `libsafetensors_c.a`) and **MSVC** (`x86_64-pc-windows-msvc`, producing `safetensors_c.lib` and copying `libsafetensors_c.a` into `dist/lib/`).
  - Includes Windows UTF-16 wide-path functions (`safetensors_open_file_w` and `safetensors_save_to_file_w`) for native `wchar_t*` / `LPCWSTR` file paths.
  - Automatically emits `#pragma comment(lib, ...)` in `safetensors.h` under MSVC for the required Windows system libraries (`ws2_32`, `userenv`, `bcrypt`, `ntdll`, `advapi32`).
- **Panic-Safe FFI Boundary**: Every `extern "C"` entry point catches unwinding panics via `std::panic::catch_unwind` and records detailed diagnostics retrievable via thread-local `safetensors_last_error_message()`.

## Directory Structure

```text
safetensors-c/
├── Cargo.toml             # Rust staticlib + cdylib configuration wrapping safetensors 0.8
├── cbindgen.toml          # cbindgen configuration for automatic C header generation
├── build.rs               # Emits platform link flags and refreshes cbindgen header
├── include/
│   └── safetensors.h      # Embeddable C99 / C11 / C++ header file
├── src/
│   └── lib.rs             # Rust FFI implementation wrapping safetensors
├── examples/
│   └── example.c          # Complete C usage example (serialize + mmap + zero-copy read)
├── CMakeLists.txt         # Cross-platform CMake build (Linux, macOS, Windows)
├── Makefile               # GNU Make build script
├── build.sh               # Linux / macOS / MSYS2 build script -> ./dist/
├── build.bat              # Windows Command Prompt build script -> .\dist\
└── build.ps1              # Windows PowerShell build script -> .\dist\
```

## Building the Library (`safetensors.h` + `.a`)

### Linux & macOS

```bash
./build.sh
```

After building, your embeddable artifacts are placed in `dist/`:
- `dist/include/safetensors.h`
- `dist/lib/libsafetensors_c.a`

### Windows (MinGW-w64 / GCC / Clang — Native `.a` Archive)

```cmd
rustup target add x86_64-pc-windows-gnu
build.bat x86_64-pc-windows-gnu
```

Produces:
- `dist\include\safetensors.h`
- `dist\lib\libsafetensors_c.a`

Link into your C project with MinGW `gcc` or `clang`:

```cmd
gcc -O2 -I./dist/include examples/example.c ./dist/lib/libsafetensors_c.a ^
    -lws2_32 -luserenv -lbcrypt -lntdll -ladvapi32 -o example.exe
```

### Windows (MSVC `cl.exe` / Visual Studio)

```cmd
build.bat x86_64-pc-windows-msvc
```

Produces both `dist\lib\safetensors_c.lib` and `dist\lib\libsafetensors_c.a` alongside `dist\include\safetensors.h`. Because `safetensors.h` includes `#pragma comment(lib, ...)` for the Windows system libraries needed by Rust's standard library, compiling with `cl.exe` is a single command:

```cmd
cl /O2 /I dist\include examples\example.c dist\lib\safetensors_c.lib
```

### CMake Integration (Linux, macOS, Windows)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Or in your own `CMakeLists.txt`:

```cmake
add_subdirectory(safetensors-c)
target_link_libraries(your_c_target PRIVATE safetensors_c)
```
