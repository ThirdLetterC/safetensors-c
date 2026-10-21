use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=include/safetensors.h");

    // Emit platform-specific linker hints for downstream build tools
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        // Required Windows system libraries when statically linking Rust's std into C/C++
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=userenv");
        println!("cargo:rustc-link-lib=bcrypt");
        println!("cargo:rustc-link-lib=ntdll");
        println!("cargo:rustc-link-lib=advapi32");
    }

    #[cfg(feature = "regen-header")]
    {
        let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let out_path = PathBuf::from(&crate_dir)
            .join("target")
            .join("include")
            .join("safetensors_cbindgen.h");

        if let Some(parent) = out_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let config = cbindgen::Config::from_root_or_default(&crate_dir);
        if let Ok(bindings) = cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_config(config)
            .generate()
        {
            let _ = bindings.write_to_file(&out_path);
        }
    }
}
