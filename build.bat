@echo off
REM Builds safetensors-c on Windows (supports both GNU/MinGW .a and MSVC .lib/.a)
REM Usage:
REM   build.bat                  (builds with active default Rust toolchain)
REM   build.bat x86_64-pc-windows-gnu   (builds native MinGW .a archive)
REM   build.bat x86_64-pc-windows-msvc  (builds MSVC static library and copies to .a + .lib)

setlocal enabledelayedexpansion
cd /d "%~dp0"

set TARGET=%~1
if "%TARGET%"=="" (
    echo ==> Building safetensors-c [default host target]...
    cargo build --release
    if errorlevel 1 exit /b 1
    set OUT_DIR=target\release
) else (
    echo ==> Building safetensors-c [target: %TARGET%]...
    cargo build --release --target %TARGET%
    if errorlevel 1 exit /b 1
    set OUT_DIR=target\%TARGET%\release
)

if not exist dist\include mkdir dist\include
if not exist dist\lib mkdir dist\lib

copy /Y include\safetensors.h dist\include\safetensors.h >nul

if exist "%OUT_DIR%\libsafetensors_c.a" (
    copy /Y "%OUT_DIR%\libsafetensors_c.a" dist\lib\libsafetensors_c.a >nul
)
if exist "%OUT_DIR%\safetensors_c.lib" (
    copy /Y "%OUT_DIR%\safetensors_c.lib" dist\lib\safetensors_c.lib >nul
    if not exist "dist\lib\libsafetensors_c.a" (
        copy /Y "%OUT_DIR%\safetensors_c.lib" dist\lib\libsafetensors_c.a >nul
    )
)

echo ==> Build complete! Embeddable artifacts in dist\:
dir /b dist\include dist\lib
endlocal
