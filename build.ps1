param(
    [string]$Target = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ([string]::IsNullOrEmpty($Target)) {
    Write-Host "==> Building safetensors-c (default host target)..."
    cargo build --release
    $OutDir = Join-Path "target" "release"
} else {
    Write-Host "==> Building safetensors-c (target: $Target)..."
    cargo build --release --target $Target
    $OutDir = Join-Path "target" (Join-Path $Target "release")
}

New-Item -ItemType Directory -Force -Path "dist/include" | Out-Null
New-Item -ItemType Directory -Force -Path "dist/lib" | Out-Null

Copy-Item -Force "include/safetensors.h" "dist/include/safetensors.h"

$LibA = Join-Path $OutDir "libsafetensors_c.a"
$LibMsvc = Join-Path $OutDir "safetensors_c.lib"

if (Test-Path $LibA) {
    Copy-Item -Force $LibA "dist/lib/libsafetensors_c.a"
}
if (Test-Path $LibMsvc) {
    Copy-Item -Force $LibMsvc "dist/lib/safetensors_c.lib"
    if (-not (Test-Path "dist/lib/libsafetensors_c.a")) {
        Copy-Item -Force $LibMsvc "dist/lib/libsafetensors_c.a"
    }
}

Write-Host "==> Done! Staged header and static archive in ./dist/"
Get-ChildItem -Recurse "dist"
