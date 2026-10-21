#!/usr/bin/env bash
# Builds safetensors-c and stages the embeddable header + static library (.a) into ./dist/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="${1:-}"
CargoArgs=("--release")
if [[ -n "$TARGET" ]]; then
    CargoArgs+=("--target" "$TARGET")
    TARGET_DIR="target/$TARGET/release"
else
    TARGET_DIR="target/release"
fi

echo "==> Building safetensors-c via cargo (${CargoArgs[*]})..."
cargo build "${CargoArgs[@]}"

mkdir -p dist/include dist/lib
cp -f include/safetensors.h dist/include/safetensors.h

if [[ -f "$TARGET_DIR/libsafetensors_c.a" ]]; then
    cp -f "$TARGET_DIR/libsafetensors_c.a" dist/lib/libsafetensors_c.a
elif [[ -f "$TARGET_DIR/safetensors_c.lib" ]]; then
    cp -f "$TARGET_DIR/safetensors_c.lib" dist/lib/safetensors_c.lib
    cp -f "$TARGET_DIR/safetensors_c.lib" dist/lib/libsafetensors_c.a
fi

echo "==> Build complete! Artifacts staged in ./dist/:"
ls -lh dist/include dist/lib
